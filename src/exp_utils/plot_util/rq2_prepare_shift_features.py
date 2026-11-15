"""
Compute semantic and textural distances for saved RQ2/RQ5 metadata pairs.
"""
import os
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import clip

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from exp_utils.dataset_process_util.emp_null_sample import (
    prepare_datasets,
    load_exclude_indices
)
from exp_utils.config_util.attack_config import attack_args, HOLD_OUT_OUTPUT_DIR, OUTPUT_ROOT_PATH
from exp_utils.load_util.load_data import create_dataset_dataloader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calc_semantic_shift(emb_a, emb_b):
    """Compute delta_sem = 2 * abs(2 * ACC - 1) from a linear classifier."""
    X = np.vstack([emb_a, emb_b])
    y = np.hstack([np.ones(len(emb_a)), np.zeros(len(emb_b))])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, shuffle=True, stratify=y
    )
    
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_te = torch.FloatTensor(X_test).to(DEVICE)
    y_te = torch.FloatTensor(y_test).to(DEVICE)
    
    model = nn.Sequential(
        nn.Linear(X.shape[1], 1),
        nn.Sigmoid()
    ).to(DEVICE)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    model.train()
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(X_tr).squeeze()
        loss = criterion(output, y_tr)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_te).squeeze()
        y_pred = (y_pred_proba > 0.5).float()
        acc = (y_pred == y_te).float().mean().item()
    
    delta_sem = 2 * abs(2 * acc - 1)
    return delta_sem, acc

def calc_single_texture_overlap(feat_a, feat_b):
    """Compute the histogram overlap of one texture feature."""
    bins = np.linspace(min(feat_a.min(), feat_b.min()), max(feat_a.max(), feat_b.max()), 50)
    hist1, _ = np.histogram(feat_a, bins=bins, density=True)
    hist2, _ = np.histogram(feat_b, bins=bins, density=True)
    overlap = np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0])
    return overlap

def calc_texture_shift(lap_a, lap_b, fft_a=None, fft_b=None):
    """
    Compute delta_txt as one minus the smaller Laplacian and FFT overlap.
    """

    overlap_lap = calc_single_texture_overlap(lap_a, lap_b)

    overlap_fft = 1.0
    if fft_a is not None and fft_b is not None:
        overlap_fft = calc_single_texture_overlap(fft_a, fft_b)

    min_overlap = min(overlap_lap, overlap_fft)

    delta_txt = 1.0 - min_overlap
    
    return delta_txt

def load_base_pair_metadata(pair_idx, base_dataset_path):
    """
    Load source dataset names and original indices for a null pair.
    """
    base1_file = os.path.join(base_dataset_path, f"pair_{pair_idx}_base1.json")
    base2_file = os.path.join(base_dataset_path, f"pair_{pair_idx}_base2.json")
    
    if not os.path.exists(base1_file) or not os.path.exists(base2_file):
        return None
    
    with open(base1_file, 'r', encoding='utf-8') as f:
        base1_meta = json.load(f)
    with open(base2_file, 'r', encoding='utf-8') as f:
        base2_meta = json.load(f)
    
    return base1_meta, base2_meta

def get_features_from_prepared_df(meta_list, prepared_df):
    """
    Resolve metadata entries to their saved image features.
    """
    features_list = []
    for item in meta_list:
        src_ds = item['source_dataset']
        orig_idx = item['original_index']

        orig_idx_type = type(orig_idx)
        df_idx_type = prepared_df['original_index'].dtype

        orig_idx_int = int(orig_idx)
        prepared_df_copy = prepared_df.copy()
        prepared_df_copy['original_index'] = prepared_df_copy['original_index'].astype(int)

        match = prepared_df_copy[
            (prepared_df_copy['source_dataset'] == src_ds) & 
            (prepared_df_copy['original_index'] == orig_idx_int)
        ]
        
        if len(match) == 0:

            print(f"Debug info:")
            print(f"  Looking for: source_dataset={src_ds}, original_index={orig_idx} (type: {orig_idx_type})")
            print(f"  DataFrame unique source_datasets: {prepared_df['source_dataset'].unique()}")
            print(f"  DataFrame original_index dtype: {df_idx_type}")
            print(f"  DataFrame sample original_index values: {prepared_df['original_index'].head(10).tolist()}")
            raise ValueError(f"Could not find sample {src_ds}_{orig_idx} in prepared_df!")

        original_match = prepared_df.iloc[match.index]
        features_list.append(original_match.iloc[0].to_dict())
    
    return pd.DataFrame(features_list)

def save_delta_features(delta_df, delta_outputs):
    """
    Save each measured pair in its corresponding RQ output directory.
    """
    for path in delta_outputs.values():
        if path.exists():
            raise FileExistsError(path)
    for case, path in delta_outputs.items():
        lower, upper = (1, 500) if case == 'rq2' else (501, 1500)
        measured = delta_df[delta_df['pair_id'].between(lower, upper)].reset_index(drop=True)
        if measured.empty:
            raise ValueError(f'No measured pairs for {case}')
        path.parent.mkdir(parents=True, exist_ok=True)
        measured.to_pickle(path)


def process_delta_features(rq2_root, output_dir, num_pairs):
    delta_outputs = {
        'rq2': Path(OUTPUT_ROOT_PATH, 'experiments', 'RQ2_hyperparameters', 'delta_features.pkl'),
        'rq5': Path(OUTPUT_ROOT_PATH, 'experiments', 'RQ5_distribution_bound_ablation', 'delta_features.pkl')
    }

    base_dataset_path = os.path.join(rq2_root, "datasets", "D_base")
    
    if not os.path.exists(os.path.join(rq2_root, "intermediate", "prepared_df.pkl")):

        print("\n[Step 1] Reloading and computing all features (like emp_null_sample.py)...")
        
        tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(attack_args.base_model_id, "tokenizer"), use_fast=False)

        print("Loading exclude indices...")
        exclude_indices = load_exclude_indices()
        
        configs = {
            'coco_2017_val': {
                "dataset": "coco_2017_val",
                "load_path": f"{attack_args.local_dataset_path}/coco_2017_val/data/val",
                "load_num": 3000,
                "split": "val",
            },
            'nocaps': {
                "dataset": "nocaps",
                "load_path": f"{attack_args.local_dataset_path}/nocaps/data/val",
                "load_num": 3000,
                "split": "val",
            },
            'laion2b_multi_translated': {
                "dataset": "laion2b_multi_translated",
                "load_path": f"{attack_args.local_dataset_path}/laion2b_multi_translated/data/val",
                "load_num": 3000,
                "split": "val",
            }
        }
        
        dataloaders = {}
        for name, cfg in configs.items():
            try:
                dl, _ = create_dataset_dataloader(
                    HOLD_OUT_OUTPUT_DIR,
                    cfg,
                    tokenizer,
                    224,
                    64,
                    need_save=False,
                    exclude_indices=exclude_indices
                )
                dataloaders[name] = dl
            except Exception as e:
                raise
        
        if not dataloaders:
            print("Error: No dataloaders could be created!")
            return None
        
        print("Loading CLIP model...")
        clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        clip_model.eval()
        
        print("Preparing datasets (computing all features)...")
    
        prepared_df = prepare_datasets(dataloaders, clip_model=clip_model, exclude_indices=exclude_indices)

        prepared_df.to_pickle(os.path.join(output_dir, "prepared_df.pkl"))
    else:

        prepared_df = pd.read_pickle(os.path.join(rq2_root, "intermediate", "prepared_df.pkl"))

        prepared_df['original_index'] = prepared_df['original_index'].astype(int)

    print("\n[Step 2] Loading base12 pairs and computing deltas...")
    
    if not all(path.exists() for path in delta_outputs.values()):
        delta_data = []
        for pair_idx in tqdm(range(1, num_pairs + 1), desc=f"Processing pairs"):
            try:
                pair_data = load_base_pair_metadata(pair_idx, base_dataset_path)
                if pair_data is None:
                    continue
                
                base1_meta, base2_meta = pair_data

                df_a = get_features_from_prepared_df(base1_meta, prepared_df)
                df_b = get_features_from_prepared_df(base2_meta, prepared_df)

                emb1 = np.array([item['embedding'] for _, item in df_a.iterrows()])
                emb2 = np.array([item['embedding'] for _, item in df_b.iterrows()])
                lap1 = np.array([item['laplacian'] for _, item in df_a.iterrows()])
                lap2 = np.array([item['laplacian'] for _, item in df_b.iterrows()])
                fft1 = np.array([item['fft'] for _, item in df_a.iterrows()])
                fft2 = np.array([item['fft'] for _, item in df_b.iterrows()])

                delta_sem, acc = calc_semantic_shift(emb1, emb2)
                delta_txt = calc_texture_shift(lap1, lap2, fft1, fft2)
                
                delta_data.append({
                    'pair_id': pair_idx,
                    'delta_sem': delta_sem,
                    'delta_txt': delta_txt,
                    'clf_acc': acc
                })
                
                print(f"Pair {pair_idx}: delta_sem={delta_sem:.4f}, delta_txt={delta_txt:.4f}, acc={acc:.4f}")
            except Exception as e:
                raise
    
        delta_df = pd.DataFrame(delta_data) if delta_data else None

        save_delta_features(delta_df, delta_outputs)
    
    else:

        delta_df = pd.concat([pd.read_pickle(path) for path in delta_outputs.values()], ignore_index=True)
    
    if delta_df is None or len(delta_df) == 0:
        print("Error: No valid pairs processed!")
        return None

def main():
    if not attack_args.rq2_root:
        raise ValueError('--rq2_root must contain the saved dataset and intermediate directories')
    output_dir = os.path.join(OUTPUT_ROOT_PATH, 'experiments', 'shared_res', 'RQ2_RQ5_calibration_pool', 'intermediate')
    os.makedirs(output_dir, exist_ok=True)
    num_pairs = len(list(Path(attack_args.rq2_root, 'datasets', 'D_base').glob('*_base1.json')))
    process_delta_features(attack_args.rq2_root, output_dir, num_pairs)

if __name__ == '__main__':
    main()
