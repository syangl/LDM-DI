"""
Run calibration and evaluation tests at the eight RQ6 dataset sizes.
"""
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from exp_utils.config_util.attack_config import attack_args, OUTPUT_ROOT_PATH
from exp_utils.dataset_process_util.meta_feature_manager import create_loader
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind
import json
import datetime
import time

def feature_norm(features):
    features_mean = features.mean(dim=0, keepdim=True)
    features_std = features.std(dim=0, keepdim=True)
    return (features - features_mean) / (features_std + 1e-8)

def handle_outliers(features, ratio=0.025):
    X = features.clone()
    for j in range(X.shape[1]):
        col = X[:, j]
        lower = torch.quantile(col, ratio)
        upper = torch.quantile(col, 1 - ratio)
        mean_val = col.mean()
        mask = (col < lower) | (col > upper)
        col = torch.where(mask, mean_val, col)
        X[:, j] = col
    return X

def process_features(sus_feats_subset, hold_out_feats_subset):
    all_features = torch.cat((sus_feats_subset, hold_out_feats_subset), dim=0)
    all_features_norm = feature_norm(all_features)

    sus_features_norm = all_features_norm[:sus_feats_subset.shape[0], :]
    hold_out_features_norm = all_features_norm[sus_feats_subset.shape[0]:, :]

    sus_features_norm = handle_outliers(sus_features_norm)
    hold_out_features_norm = handle_outliers(hold_out_features_norm)

    sus_label = torch.ones(sus_features_norm.shape[0]).unsqueeze(1)
    hold_out_label = torch.zeros(hold_out_features_norm.shape[0]).unsqueeze(1)

    return sus_features_norm, sus_label, hold_out_features_norm, hold_out_label

def train_model(sus_features, sus_label, hold_out_features, hold_out_label, pair_idx, test_i, MODEL_OUTPUT_DIR):
    model = torch.nn.Sequential(
        torch.nn.Linear(sus_features.shape[1], 1),
    )
    save_file = os.path.join(MODEL_OUTPUT_DIR, f"model_{pair_idx}_{test_i}.pt")
    if os.path.exists(save_file):
        model.load_state_dict(torch.load(save_file))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    all_features = torch.cat((sus_features, hold_out_features), dim=0)
    all_labels = torch.cat((sus_label, hold_out_label), dim=0)
    total_num = all_features.size(0)

    epochs = 1000
    model.train()
    for epoch in range(epochs):
        indices = torch.randperm(total_num)
        tmp_features = all_features[indices]
        tmp_labels = all_labels[indices]

        optimizer.zero_grad()
        outputs = model(tmp_features)
        loss = criterion(outputs, tmp_labels)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_file)
    return model

def predict(model, features):
    model.eval()
    logits = model(features).detach()
    return torch.sigmoid(logits)

def t_test(B_sus, B_hold_out):
    t, _ = ttest_ind(B_sus, B_hold_out, equal_var=False, alternative='greater')
    return t.item()

def is_hold_dataset(ds):
    hold_datasets = ["coco_2017_val", "laion2b_multi_translated", "nocaps"]
    return ds.startswith("D_hold_out_") or ds in hold_datasets

def is_hold_hold_pair(ds1, ds2):
    return is_hold_dataset(ds1) and is_hold_dataset(ds2)

def is_sus_hold_pair(ds1, ds2):
    has_sus = (not is_hold_dataset(ds1) and ds1 not in ["D_base_1", "D_base_2"])
    has_sus = has_sus or (not is_hold_dataset(ds2) and ds2 not in ["D_base_1", "D_base_2"])
    has_hold = is_hold_dataset(ds1) or is_hold_dataset(ds2)
    return has_sus and has_hold

def is_base12_pair(ds1, ds2):
    return {ds1, ds2} == {"D_base_1", "D_base_2"}

def load_t_stats_from_dir(ttest_dir, k_base=None):
    """
    Load t-statistics from a directory
    """
    all_t_stats = []
    dataset_info_list = []
    
    if not os.path.exists(ttest_dir):
        return np.array([]), []
    
    pair_dirs = [d for d in os.listdir(ttest_dir) if d.startswith("pair_") and os.path.isdir(os.path.join(ttest_dir, d))]
    pair_dirs.sort()
    
    base12_pair_dirs = []
    other_pair_dirs = []
    
    for pair_dir in pair_dirs:
        pair_idx = pair_dir.split("_")[1]
        if pair_idx.isdigit():
            base12_pair_dirs.append(pair_dir)
        else:
            other_pair_dirs.append(pair_dir)
    
    if k_base is not None:
        import random
        random.seed(0)
        base12_pair_dirs = random.sample(base12_pair_dirs, min(k_base, len(base12_pair_dirs)))
    
    selected_pair_dirs = base12_pair_dirs + other_pair_dirs
    
    for pair_dir in selected_pair_dirs:
        pair_idx = pair_dir.split("_")[1]
        ttest_file_path = os.path.join(ttest_dir, pair_dir, f"ttest_pair_{pair_idx}_results.json")
        
        if not os.path.exists(ttest_file_path):
            continue
            
        with open(ttest_file_path, "r", encoding='utf-8') as f:
            results = json.load(f)
            for result in results:
                all_t_stats.append(result['t_statistic'])
                dataset_info = {
                    "dataset1": result.get('dataset1', 'unknown'),
                    "dataset2": result.get('dataset2', 'unknown'),
                    "t_statistic": result['t_statistic']
                }
                dataset_info_list.append(dataset_info)
    
    return np.array(all_t_stats), dataset_info_list

def calculate_rejection_rate(t_stats, tau):
    """
    Calculate rejection rate: proportion of t-statistics greater than tau
    """
    num_rejections = np.sum(t_stats > tau)
    rejection_rate = num_rejections / len(t_stats) if len(t_stats) > 0 else 0.0
    return rejection_rate, num_rejections

def empirical_quantile_mean(base_t_stats, alpha, trim_percent=0.0):
    """
    Empirical quantile threshold calibration with outlier replacement using mean.
    """
    stats = np.copy(base_t_stats)
    
    if trim_percent is not None and trim_percent > 0.0:
        lower = np.percentile(stats, 100 * trim_percent)
        upper = np.percentile(stats, 100 * (1 - trim_percent))
        mean_val = np.mean(stats)
        mask = (stats < lower) | (stats > upper)
        cleaned_stats = np.where(mask, mean_val, stats)
    else:
        cleaned_stats = stats
    
    q = 100 * (1 - alpha)
    tau_quantile = np.percentile(cleaned_stats, q)
    
    return {
        "tau_quantile": float(tau_quantile),
        "alpha": alpha,
    }

def calculate_tpr_fpr_from_ttest_dir(ttest_dir, alpha=0.001, k_base=None):
    """
    Calculate TPR and FPR using the same logic as calculate_threshold_fpr.py
    """
    all_t_stats, all_dataset_info = load_t_stats_from_dir(ttest_dir, k_base=k_base)
    
    if len(all_t_stats) == 0:
        return None
    
    base12_t_stats = []
    hold_hold_t_stats = []
    sus_hold_t_stats = []
    base12_info = None
    hold_hold_info = None
    sus_hold_info = None
    
    for info in all_dataset_info:
        ds1, ds2 = info['dataset1'], info['dataset2']
        t = info['t_statistic']
        if is_base12_pair(ds1, ds2):
            base12_t_stats.append(t)
            if base12_info is None:
                base12_info = {"dataset1": ds1, "dataset2": ds2}
        elif is_hold_hold_pair(ds1, ds2):
            hold_hold_t_stats.append(t)
            if hold_hold_info is None:
                hold_hold_info = {"dataset1": ds1, "dataset2": ds2}
        elif is_sus_hold_pair(ds1, ds2):
            sus_hold_t_stats.append(t)
            if sus_hold_info is None:
                sus_hold_info = {"dataset1": ds1, "dataset2": ds2}
    
    base12_t_stats = np.array(base12_t_stats)
    hold_hold_t_stats = np.array(hold_hold_t_stats)
    sus_hold_t_stats = np.array(sus_hold_t_stats)
    
    if len(base12_t_stats) == 0:
        return None
    
    quantile_result = empirical_quantile_mean(base12_t_stats, alpha=alpha)
    tau = quantile_result["tau_quantile"]
    
    fpr = None
    num_false_rejections = None
    if len(hold_hold_t_stats) > 0:
        fpr, num_false_rejections = calculate_rejection_rate(hold_hold_t_stats, tau)
    
    tpr = None
    num_true_rejections = None
    if len(sus_hold_t_stats) > 0:
        tpr, num_true_rejections = calculate_rejection_rate(sus_hold_t_stats, tau)
    
    return {
        "tau": tau,
        "alpha": alpha,
        "base12": {
            "num_t_stats": len(base12_t_stats),
            "mean_t": float(np.mean(base12_t_stats)),
            "std_t": float(np.std(base12_t_stats)),
            "min_t": float(np.min(base12_t_stats)),
            "max_t": float(np.max(base12_t_stats)),
        },
        "fpr": {
            "tpr_fpr": fpr,
            "num_false_rejections": num_false_rejections,
            "num_tests": len(hold_hold_t_stats),
        },
        "tpr": {
            "tpr": tpr,
            "num_true_rejections": num_true_rejections,
            "num_tests": len(sus_hold_t_stats),
        }
    }

def process_single_pair(loader, pair_idx, dataset1_name, dataset2_name, MODEL_OUTPUT_DIR, TTest_OUTPUT_DIR, SPLIT_TIMES, SPLIT_DS_LEN, dataset_total_num):
    """
    Process a single dataset pair for t-test with specified dataset size
    """
    ttest_file_path = os.path.join(TTest_OUTPUT_DIR, f"pair_{pair_idx}", f"ttest_pair_{pair_idx}_results.json")
    
    if os.path.exists(ttest_file_path):
        with open(ttest_file_path, "r", encoding='utf-8') as f:
            return json.load(f)
    
    print(f"  Pair {pair_idx}: {dataset1_name} vs {dataset2_name}")
    
    ds1_feats, ds2_feats = loader.load_dataset_pair_features(dataset1_name, dataset2_name, pair_idx)
    
    if dataset_total_num < ds1_feats.shape[0]:
        target_size = dataset_total_num
        sampler = torch.Generator().manual_seed(0)
        sampled_indices = torch.randperm(ds1_feats.shape[0], generator=sampler)[:target_size]
        ds1_feats = ds1_feats[sampled_indices]
        ds2_feats = ds2_feats[sampled_indices]
    
    total_ds_len = ds1_feats.shape[0]
    results = []
    
    for test_i in tqdm(range(SPLIT_TIMES), desc="Running t-test for size..."):
        torch.manual_seed(attack_args.global_seed + test_i)
        test_start = time.perf_counter()
        preds_file_path = os.path.join(TTest_OUTPUT_DIR, f"pair_{pair_idx}", f"preds_{test_i}.json")
        
        generator = torch.Generator().manual_seed(test_i)
        indices = torch.randperm(total_ds_len, generator=generator)
        A_indices = indices[:SPLIT_DS_LEN]
        B_indices = indices[SPLIT_DS_LEN:]
        
        features_A_tuple = process_features(ds1_feats[A_indices], ds2_feats[A_indices])
        features_B_tuple = process_features(ds1_feats[B_indices], ds2_feats[B_indices])
        
        model = train_model(
            features_A_tuple[0], features_A_tuple[1],
            features_A_tuple[2], features_A_tuple[3],
            pair_idx, test_i, MODEL_OUTPUT_DIR
        )
        
        features_B = torch.cat((features_B_tuple[0], features_B_tuple[2]), dim=0)
        label_B = torch.cat((features_B_tuple[1], features_B_tuple[3]), dim=0)
        
        preds = predict(model, features_B)
        B_sus = preds[label_B == 1].cpu().numpy()
        B_hold_out = preds[label_B == 0].cpu().numpy()
        
        os.makedirs(os.path.dirname(preds_file_path), exist_ok=True)
        with open(preds_file_path, "w", encoding='utf-8') as f:
            json.dump({
                "B_sus_preds": B_sus.tolist(),
                "B_hold_out_preds": B_hold_out.tolist(),
                "dataset_total_num": dataset_total_num,
            }, f, indent=4)
        
        t_statistic = t_test(B_sus, B_hold_out)
        
        results.append({
            "pair_idx": pair_idx,
            "dataset1": dataset1_name,
            "dataset2": dataset2_name,
            "test_i": test_i,
            "t_statistic": t_statistic,
            "test_elapsed_seconds": time.perf_counter() - test_start,
        })
    
    with open(ttest_file_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    return results

def run_ttest_for_size(dataset_size, SPLIT_TIMES=100):
    """
    Run t-test for a specific dataset size
    """
    attack_args.dataset_total_num = dataset_size
    SPLIT_DS_LEN = dataset_size // 2
    
    dataset1_name = attack_args.used_datasets[0]
    dataset2_name = attack_args.used_datasets[1]
    
    is_base12_pair = {dataset1_name, dataset2_name} == {"D_base_1", "D_base_2"}
    
    ROOT_OUTPUT = os.path.join(OUTPUT_ROOT_PATH, "experiments", "RQ6_dataset_size", "runs", str(dataset_size))
    MODEL_OUTPUT_DIR = os.path.join(ROOT_OUTPUT, "model")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    TTest_OUTPUT_DIR = os.path.join(ROOT_OUTPUT, "results")
    os.makedirs(TTest_OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running with dataset_size={dataset_size}")
    print(f"Output dir: {TTest_OUTPUT_DIR}")
    print(f"{'='*60}")
    
    loader = create_loader()
    all_results = []
    
    if is_base12_pair:
        K_BASE = attack_args.k_base
        for pair_idx in range(K_BASE):
            try:
                results = process_single_pair(
                    loader, pair_idx + 1, dataset1_name, dataset2_name,
                    MODEL_OUTPUT_DIR, TTest_OUTPUT_DIR, 1, SPLIT_DS_LEN, dataset_size
                )
                all_results.extend(results)
            except FileNotFoundError:
                raise
    else:
        if is_hold_hold_pair(dataset1_name, dataset2_name):
            pair_idx = "holdhold"
        elif is_sus_hold_pair(dataset1_name, dataset2_name):
            pair_idx = "sushold"
        else:
            raise ValueError(f"Unknown dataset pair type: {dataset1_name} vs {dataset2_name}")
        
        results = process_single_pair(
            loader, pair_idx, dataset1_name, dataset2_name,
            MODEL_OUTPUT_DIR, TTest_OUTPUT_DIR, SPLIT_TIMES, SPLIT_DS_LEN, dataset_size
        )
        all_results.extend(results)
    
    return all_results, TTest_OUTPUT_DIR

if __name__ == "__main__":
    DATASET_SIZES = attack_args.dataset_sizes
    SPLIT_TIMES = attack_args.split_times
    ALPHA = 0.001
    K_BASE = attack_args.k_base
    
    dataset1_name = attack_args.used_datasets[0]
    dataset2_name = attack_args.used_datasets[1]
    model_name = attack_args.model_name
    
    overall_start_time = time.time()
    
    all_size_results = {}
    
    for size in DATASET_SIZES:
        size_start_time = time.time()
        
        ROOT_OUTPUT = os.path.join(OUTPUT_ROOT_PATH, "experiments", "RQ6_dataset_size", "runs", str(size))
        TTest_OUTPUT_DIR = os.path.join(ROOT_OUTPUT, "results")
        
        all_results, ttest_dir = run_ttest_for_size(size, SPLIT_TIMES)

        size_elapsed = time.time() - size_start_time
        
        t_stats = [r['t_statistic'] for r in all_results]
        
        all_size_results[size] = {
            "size": size,
            "ttest_dir": ttest_dir,
            "num_tests": len(all_results),
            "mean_t": float(np.mean(t_stats)),
            "std_t": float(np.std(t_stats)),
            "min_t": float(np.min(t_stats)),
            "max_t": float(np.max(t_stats)),
            "elapsed_seconds": size_elapsed,
            "t_statistics": t_stats,
            "results": all_results
        }
        
        print(f"  Size {size}: Mean t={all_size_results[size]['mean_t']:.4f}, Std={all_size_results[size]['std_t']:.4f}")
    
    overall_elapsed = time.time() - overall_start_time
    
    summary_data = {
        "model_name": model_name,
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        "split_times": SPLIT_TIMES,
        "alpha": ALPHA,
        "k_base": K_BASE,
        "dataset_sizes": DATASET_SIZES,
        "start_time": datetime.datetime.now().isoformat(),
        "results": all_size_results
    }
    
    summary_file = os.path.join(OUTPUT_ROOT_PATH, "experiments", "RQ6_dataset_size", "runs", "loop_ttest_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*80)
    print(f"{'Dataset Size':<15} {'#Tests':<10} {'Mean t':<12} {'Std t':<12} {'TPR':<12} {'FPR':<12} {'tau':<12}")
    print("="*80)
    
    for size in DATASET_SIZES:
        r = all_size_results[size]
        tpr_val = r['tpr_fpr']['tpr']['tpr'] if 'tpr_fpr' in r else None
        fpr_val = r['tpr_fpr']['fpr']['tpr_fpr'] if 'tpr_fpr' in r else None
        tau_val = r['tpr_fpr']['tau'] if 'tpr_fpr' in r else None
        
        tpr_str = f"{tpr_val:.4f}" if tpr_val is not None else "N/A"
        fpr_str = f"{fpr_val:.4f}" if fpr_val is not None else "N/A"
        tau_str = f"{tau_val:.4f}" if tau_val is not None else "N/A"
        
        print(f"{size:<15} {r['num_tests']:<10} {r['mean_t']:<12.4f} {r['std_t']:<12.4f} {tpr_str:<12} {fpr_str:<12} {tau_str:<12}")
    
    print("="*80)
    print(f"\nTotal elapsed time: {overall_elapsed:.2f}s ({overall_elapsed/3600:.2f}h)")
    print(f"Summary saved to: {summary_file}")
    print("="*80)
