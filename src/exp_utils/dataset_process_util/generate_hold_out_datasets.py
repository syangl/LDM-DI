import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import clip
from transformers import CLIPTokenizer
from exp_utils.config_util.attack_config import attack_args, HOLD_OUT_OUTPUT_DIR
from exp_utils.load_util.load_data import create_dataset_dataloader


TARGET_TOTAL_SAMPLES = 1000
LOAD_NUM = 3000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = os.path.join(attack_args.local_dataset_path)


def save_metadata_only(df, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    metadata_list = []
    for _, row in df.iterrows():
        metadata_list.append({
            'source_dataset': row['source_dataset'],
            'original_index': int(row['original_index'])
        })
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4)


def compute_metrics(imgs):
    metrics = {}
    gray = imgs.mean(dim=1, keepdim=True)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
    edges = F.conv2d(gray, kernel, padding=1)
    metrics['laplacian'] = torch.var(edges, dim=[1, 2, 3]).cpu().numpy()

    f = torch.fft.fft2(gray.squeeze(1))
    fshift = torch.fft.fftshift(f)
    magnitude = torch.abs(fshift)
    rows, cols = gray.shape[-2:]
    crow, ccol = rows // 2, cols // 2
    r = int(min(rows, cols) * 0.1)
    y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
    mask = ((x.to(DEVICE) - ccol)**2 + (y.to(DEVICE) - crow)**2) <= r**2
    total_energy = torch.sum(magnitude, dim=[1, 2])
    low_energy = torch.sum(magnitude * mask, dim=[1, 2])
    metrics['fft'] = ((total_energy - low_energy) / (total_energy + 1e-8)).cpu().numpy()

    return metrics


def prepare_datasets(dataloaders, clip_model):
    metadata_list = []
    for loader_name, loader in dataloaders.items():
        print(f"Processing {loader_name}...")
        for batch in tqdm(loader):
            imgs = batch['image']
            indices = batch['index']
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy().tolist()

            valid_imgs_tensor = imgs.to(DEVICE)
            imgs_metrics = compute_metrics(valid_imgs_tensor)

            with torch.no_grad():
                image_features = clip_model.encode_image(valid_imgs_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings = image_features.cpu().numpy()

            for i in range(len(indices)):
                metadata_list.append({
                    'source_dataset': loader_name,
                    'original_index': indices[i],
                    'laplacian': imgs_metrics['laplacian'][i],
                    'fft': imgs_metrics['fft'][i],
                    'embedding': embeddings[i],
                })
    df = pd.DataFrame(metadata_list)
    return df


def split_datasets(df):
    metric_cols = ['laplacian', 'fft']
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[metric_cols] = scaler.fit_transform(df[metric_cols])

    kmeans_txt = KMeans(n_init='auto')
    df['txt_cluster'] = kmeans_txt.fit_predict(df_norm[metric_cols])

    emb_matrix = np.stack(df['embedding'].values)
    kmeans_clip = KMeans(n_init='auto')
    df['clip_cluster'] = kmeans_clip.fit_predict(emb_matrix)

    source_map = {
        'coco_2017_val': 0.0,
        'laion2b_multi_translated': 1.0,
        'nocaps': 2.0,
    }
    df['source_val'] = df['source_dataset'].map(source_map).fillna(1.5)

    all_pairs = []
    groups = df.groupby(['txt_cluster', 'clip_cluster'])

    for _, group in groups:
        idxs = group.index.tolist()
        if len(idxs) < 2:
            continue

        scored_idxs = []
        for idx in idxs:
            val = group.loc[idx, 'source_val']
            noise = random.gauss(0, 1.0)
            scored_idxs.append((idx, val + noise))

        scored_idxs.sort(key=lambda x: x[1])
        sorted_indices = [x[0] for x in scored_idxs]

        num_pairs = len(sorted_indices) // 2
        idx_for_a = sorted_indices[:num_pairs]
        idx_for_b = sorted_indices[-num_pairs:]

        for ia, ib in zip(idx_for_a, idx_for_b):
            all_pairs.append((ia, ib))

    if len(all_pairs) < TARGET_TOTAL_SAMPLES:
        final_pairs = all_pairs
    else:
        random.shuffle(all_pairs)
        final_pairs = all_pairs[:TARGET_TOTAL_SAMPLES]

    indices_a = [p[0] for p in final_pairs]
    indices_b = [p[1] for p in final_pairs]

    df_a = df.loc[indices_a].reset_index(drop=True)
    df_b = df.loc[indices_b].reset_index(drop=True)
    return df_a, df_b


if __name__ == "__main__":
    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(attack_args.base_model_id, "tokenizer"))

    print("="*60)
    print("Generating hold-out datasets D_hold_out_1 and D_hold_out_2...")
    print("="*60)

    configs = {
        'coco_2017_val': {
            "dataset": "coco_2017_val",
            "load_path": f"{attack_args.local_dataset_path}/coco_2017_val/data/val",
            "load_num": LOAD_NUM,
            "split": "val",
        },
        'nocaps': {
            "dataset": "nocaps",
            "load_path": f"{attack_args.local_dataset_path}/nocaps/data/val",
            "load_num": LOAD_NUM,
            "split": "val",
        },
        'laion2b_multi_translated': {
            "dataset": "laion2b_multi_translated",
            "load_path": f"{attack_args.local_dataset_path}/laion2b_multi_translated/data/val",
            "load_num": LOAD_NUM,
            "split": "val",
        }
    }

    dataloaders = {}
    for name, cfg in configs.items():
        dl, _ = create_dataset_dataloader(HOLD_OUT_OUTPUT_DIR,
                                          cfg,
                                          tokenizer,
                                          224,
                                          64,
                                          need_save=False)
        dataloaders[name] = dl

    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    prepared_df = prepare_datasets(dataloaders, clip_model=clip_model)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("\n--- Splitting datasets ---")
    df_hold_out_1, df_hold_out_2 = split_datasets(prepared_df)

    print("\n--- Saving D_hold_out_1 metadata ---")
    base1_file = os.path.join(OUTPUT_ROOT, "D_hold_out_1", "metadata", "meta_D_hold_out_1.json")
    save_metadata_only(df_hold_out_1, base1_file)

    print("\n--- Saving D_hold_out_2 metadata ---")
    base2_file = os.path.join(OUTPUT_ROOT, "D_hold_out_2", "metadata", "meta_D_hold_out_2.json")
    save_metadata_only(df_hold_out_2, base2_file)

    print("\n" + "="*60)
    print("Successfully generated D_hold_out_1 and D_hold_out_2!")
    print(f"Location: {OUTPUT_ROOT}")
    print("="*60)
