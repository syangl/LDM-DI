import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import json
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import clip
from transformers import CLIPTokenizer
from exp_utils.config_util.attack_config import attack_args, HOLD_OUT_OUTPUT_DIR
from exp_utils.load_util.load_data import create_dataset_dataloader


# def determine_k(total_samples):
#     # for example, we split the dataset into 50 samples per cluster
#     samples_per_cluster = 50
#     total_clusters = total_samples / samples_per_cluster

#     # let TEXTURE_CLUSTERS = CLIP_CLUSTERS = k
#     # k^2 = total_clusters => k = sqrt(total_clusters)
#     k_txt = k_clip = int(np.sqrt(total_clusters))
#     return max(k_txt, 10), max(k_clip, 10)

TARGET_TOTAL_SAMPLES = 1000
LOAD_NUM = 3000
NUM_PAIRS = 100
# TEXTURE_CLUSTERS, CLIP_CLUSTERS = determine_k(LOAD_NUM*4)
# print(f"TEXTURE_CLUSTERS: {TEXTURE_CLUSTERS}, CLIP_CLUSTERS: {CLIP_CLUSTERS}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = os.path.join(current_dir, "mixed_datasets_metadata")



def compute_metrics(imgs):
    metrics = {}
    # Laplacian variance
    gray = imgs.mean(dim=1, keepdim=True)
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=DEVICE).view(1, 1, 3, 3)
    edges = F.conv2d(gray, kernel, padding=1)
    metrics['laplacian'] = torch.var(edges, dim=[1, 2, 3]).cpu().numpy()

    # HF proportion
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

            imgs = imgs.to(DEVICE)
            imgs_metrics = compute_metrics(imgs)

            with torch.no_grad():
                image_features = clip_model.encode_image(imgs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings = image_features.cpu().numpy()

            batch_size = imgs.shape[0]
            for i in range(batch_size):
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

    kmeans_txt = KMeans(random_state=0,
                        n_init='auto')
    df['txt_cluster'] = kmeans_txt.fit_predict(df_norm[metric_cols])

    emb_matrix = np.stack(df['embedding'].values)
    kmeans_clip = KMeans(random_state=0,
                        n_init='auto')
    df['clip_cluster'] = kmeans_clip.fit_predict(emb_matrix)

    source_map = {
        'coco2017': 0.0,
        'nocaps': 1.0,
        'laion_multi': 2.0
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


def classifier_test(df_a, df_b):
    X_a = np.stack(df_a['embedding'].values)
    X_b = np.stack(df_b['embedding'].values)
    X = np.vstack([X_a, X_b])
    y = np.hstack([np.ones(len(X_a)), np.zeros(len(X_b))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0, shuffle=True)

    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)

    model = nn.Sequential(nn.Linear(X.shape[1], 1), nn.Sigmoid()).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    model.train()
    for _ in range(1000):
        optimizer.zero_grad()
        out = model(X_train_t).squeeze()
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).squeeze()
        cls_preds = (preds > 0.5).float()
        acc = (cls_preds == y_test_t).float().mean().item()
    print(f"Classifier Accuracy: {acc:.4f}")


def verify_datasets(df_a, df_b):
    classifier_test(df_a, df_b)

    src_a = df_a['source_dataset'].value_counts(normalize=True)
    src_b = df_b['source_dataset'].value_counts(normalize=True)
    print(f"{'Source':<15} | {'Set A':<10} | {'Set B':<10} | {'Diff'}")
    all_srcs = list(set(src_a.index) | set(src_b.index))
    total_diff = 0
    for src in sorted(all_srcs):
        pa = src_a.get(src, 0.0)
        pb = src_b.get(src, 0.0)
        diff = abs(pa - pb)
        total_diff += diff
        print(f"{src:<15} | {pa:.2%} | {pb:.2%} | {diff:.2%}")
    return True


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


def save_datasets(df, folder_name, dataset_configs):
    base_save_dir = os.path.join(OUTPUT_ROOT, folder_name)
    img_save_dir = os.path.join(base_save_dir, "data", "val")
    meta_save_dir = os.path.join(base_save_dir, "metadata")

    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(meta_save_dir, exist_ok=True)

    dst_meta_file = os.path.join(meta_save_dir, f"meta_{folder_name}.jsonl")

    source_meta_cache = {}
    unique_sources = df['source_dataset'].unique()

    for source_name in unique_sources:
        config = dataset_configs[source_name]
        dataset_load_path = config['load_path']
        base_path = dataset_load_path.replace("data/val", "")
        real_dataset_name = config['dataset']

        src_meta_file = os.path.join(base_path, "metadata", f"meta_{real_dataset_name}.jsonl")

        with open(src_meta_file, 'r', encoding='utf-8') as f:
            temp_dict = {}
            for line in f:
                item = json.loads(line)
                idx_key = str(item['index'])
                temp_dict[idx_key] = item
            source_meta_cache[source_name] = temp_dict

    save_count = 0
    with open(dst_meta_file, 'w', encoding='utf-8') as dst_f:
        for new_index, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
            source_name = row['source_dataset']
            original_index = row['original_index']
            config = dataset_configs[source_name]

            src_img_path = os.path.join(config['load_path'], f"{original_index}.jpg")

            new_filename = f"{new_index}.jpg"
            dst_img_path = os.path.join(img_save_dir, new_filename)

            img_pil = Image.open(src_img_path)
            img_pil.save(dst_img_path)
            save_count += 1

            meta_dict = source_meta_cache[source_name]
            meta_item = meta_dict[str(original_index)]

            cap = meta_item['caption']
            if isinstance(cap, list):
                cap = max(cap, key=len)
            meta_item['caption'] = cap

            meta_item['index'] = new_index
            meta_item['original_index'] = original_index
            meta_item['original_source'] = source_name

            dst_f.write(json.dumps(meta_item) + '\n')



if __name__ == "__main__":
    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(attack_args.base_model_id, "tokenizer"))

    configs = {
        'coco2017': {
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
        'laion_multi': {
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

    for pair_i in tqdm(range(NUM_PAIRS), desc="Generating base12 pairs"):
        print(f"\n--- Generating pair {pair_i + 1}/{NUM_PAIRS} ---")
        mixed_df_1, mixed_df_2 = split_datasets(prepared_df)
        
        base1_file = os.path.join(OUTPUT_ROOT, f"pair_{pair_i}_base1.json")
        base2_file = os.path.join(OUTPUT_ROOT, f"pair_{pair_i}_base2.json")
        
        save_metadata_only(mixed_df_1, base1_file)
        save_metadata_only(mixed_df_2, base2_file)
        
        print(f"Saved pair {pair_i}: {base1_file}, {base2_file}")
    
    print(f"\nSuccessfully generated {NUM_PAIRS} base12 pairs in {OUTPUT_ROOT}")

