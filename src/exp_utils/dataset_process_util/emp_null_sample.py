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


def load_exclude_indices():
    exclude_indices = set()
    
    hold_out_names = ["D_hold_out_1", "D_hold_out_2"]
    for ds_name in hold_out_names:
        meta_file = os.path.join(attack_args.local_dataset_path, ds_name, "metadata", f"meta_{ds_name}.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            for item in metadata_list:
                key = f"{item['source_dataset']}_{item['original_index']}"
                exclude_indices.add(key)
            print(f"Loaded {len(metadata_list)} exclude indices from {ds_name}")
        else:
            print(f"Warning: {meta_file} not found, will not exclude samples from {ds_name}")
    
    print(f"Total exclude indices: {len(exclude_indices)}")
    return exclude_indices

TARGET_TOTAL_SAMPLES = 1000
LOAD_NUM = 3000
NUM_PAIRS = 500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_ROOT = os.path.join(attack_args.local_dataset_path, "D_base")



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



def prepare_datasets(dataloaders, clip_model, exclude_indices=None):
    if exclude_indices is None:
        exclude_indices = set()
    
    metadata_list = []
    for loader_name, loader in dataloaders.items():
        print(f"Processing {loader_name}...")
        for batch in tqdm(loader):
            imgs = batch['image']
            indices = batch['index']
            if isinstance(indices, torch.Tensor):
                indices = indices.cpu().numpy().tolist()

            valid_mask = []
            valid_imgs = []
            valid_indices = []
            
            for i in range(len(indices)):
                key = f"{loader_name}_{indices[i]}"
                if key not in exclude_indices:
                    valid_mask.append(i)
                    valid_imgs.append(imgs[i])
                    valid_indices.append(indices[i])
            
            if len(valid_imgs) == 0:
                continue
            
            valid_imgs_tensor = torch.stack(valid_imgs).to(DEVICE)
            imgs_metrics = compute_metrics(valid_imgs_tensor)

            with torch.no_grad():
                image_features = clip_model.encode_image(valid_imgs_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings = image_features.cpu().numpy()

            for i in range(len(valid_indices)):
                metadata_list.append({
                    'source_dataset': loader_name,
                    'original_index': valid_indices[i],
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


# def save_datasets(df, folder_name, dataset_configs):
#     base_save_dir = os.path.join(OUTPUT_ROOT, folder_name)
#     img_save_dir = os.path.join(base_save_dir, "data", "val")
#     meta_save_dir = os.path.join(base_save_dir, "metadata")

#     os.makedirs(img_save_dir, exist_ok=True)
#     os.makedirs(meta_save_dir, exist_ok=True)

#     dst_meta_file = os.path.join(meta_save_dir, f"meta_{folder_name}.jsonl")

#     source_meta_cache = {}
#     unique_sources = df['source_dataset'].unique()

#     for source_name in unique_sources:
#         config = dataset_configs[source_name]
#         dataset_load_path = config['load_path']
#         base_path = dataset_load_path.replace("data/val", "")
#         real_dataset_name = config['dataset']

#         src_meta_file = os.path.join(base_path, "metadata", f"meta_{real_dataset_name}.jsonl")

#         with open(src_meta_file, 'r', encoding='utf-8') as f:
#             temp_dict = {}
#             for line in f:
#                 item = json.loads(line)
#                 idx_key = str(item['index'])
#                 temp_dict[idx_key] = item
#             source_meta_cache[source_name] = temp_dict

#     save_count = 0
#     with open(dst_meta_file, 'w', encoding='utf-8') as dst_f:
#         for new_index, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df)):
#             source_name = row['source_dataset']
#             original_index = row['original_index']
#             config = dataset_configs[source_name]

#             src_img_path = os.path.join(config['load_path'], f"{original_index}.jpg")

#             new_filename = f"{new_index}.jpg"
#             dst_img_path = os.path.join(img_save_dir, new_filename)

#             img_pil = Image.open(src_img_path)
#             img_pil.save(dst_img_path)
#             save_count += 1

#             meta_dict = source_meta_cache[source_name]
#             meta_item = meta_dict[str(original_index)]

#             cap = meta_item['caption']
#             if isinstance(cap, list):
#                 cap = max(cap, key=len)
#             meta_item['caption'] = cap

#             meta_item['index'] = new_index
#             meta_item['original_index'] = original_index
#             meta_item['original_source'] = source_name

#             dst_f.write(json.dumps(meta_item) + '\n')



if __name__ == "__main__":
    tokenizer = CLIPTokenizer.from_pretrained(
        os.path.join(attack_args.base_model_id, "tokenizer"))

    print("="*60)
    print("Loading exclude indices from D_hold_out_1 and D_hold_out_2...")
    exclude_indices = load_exclude_indices()
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
                                          need_save=False,
                                          exclude_indices=exclude_indices)
        dataloaders[name] = dl

    print("Loading CLIP model...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    prepared_df = prepare_datasets(dataloaders, clip_model=clip_model, exclude_indices=exclude_indices)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for pair_i in tqdm(range(NUM_PAIRS), desc="Generating base12 pairs"):
        print(f"\n--- Generating pair {pair_i + 1}/{NUM_PAIRS} ---")
        
        base1_file = os.path.join(OUTPUT_ROOT, f"pair_{pair_i + 1}_base1.json")
        base2_file = os.path.join(OUTPUT_ROOT, f"pair_{pair_i + 1}_base2.json")
        
        if os.path.exists(base1_file) and os.path.exists(base2_file):
            print(f"Pair {pair_i + 1} already exists, skipping.")
            continue
        
        mixed_df_1, mixed_df_2 = split_datasets(prepared_df)
        
        save_metadata_only(mixed_df_1, base1_file)
        save_metadata_only(mixed_df_2, base2_file)  
        
        print(f"Saved pair {pair_i + 1}: {base1_file}, {base2_file}")
    
    print(f"\nSuccessfully generated {NUM_PAIRS} base12 pairs in {OUTPUT_ROOT}")

