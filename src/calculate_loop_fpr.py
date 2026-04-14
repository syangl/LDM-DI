import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args
import numpy as np
import json
import random


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
    """Load t-statistics from a directory"""
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
    """Calculate rejection rate: proportion of t-statistics greater than tau"""
    num_rejections = np.sum(t_stats > tau)
    rejection_rate = num_rejections / len(t_stats) if len(t_stats) > 0 else 0.0
    return rejection_rate, num_rejections


def empirical_quantile_mean(base_t_stats, alpha, trim_percent=0.0):
    """Empirical quantile threshold calibration with outlier replacement using mean."""
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
    
    return tau_quantile


def calculate_tpr_fpr_for_size(ttest_dir, alpha=0.001, k_base=400):
    """Calculate TPR and FPR for a specific dataset size directory"""
    all_t_stats, all_dataset_info = load_t_stats_from_dir(ttest_dir, k_base=k_base)
    
    if len(all_t_stats) == 0:
        print(f"Warning: No t-stats found in {ttest_dir}")
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
        print(f"Warning: No base12 data in {ttest_dir}")
        return None
    
    tau = empirical_quantile_mean(base12_t_stats, alpha=alpha)
    
    fpr = None
    num_false_rejections = None
    if len(hold_hold_t_stats) > 0:
        fpr, num_false_rejections = calculate_rejection_rate(hold_hold_t_stats, tau)
    
    tpr = None
    num_true_rejections = None
    if len(sus_hold_t_stats) > 0:
        tpr, num_true_rejections = calculate_rejection_rate(sus_hold_t_stats, tau)
    
    return {
        "tau": float(tau),
        "alpha": alpha,
        "base12": {
            "num_t_stats": int(len(base12_t_stats)),
            "mean_t": float(np.mean(base12_t_stats)),
            "std_t": float(np.std(base12_t_stats)),
            "min_t": float(np.min(base12_t_stats)),
            "max_t": float(np.max(base12_t_stats)),
        },
        "fpr": {
            "value": float(fpr) if fpr is not None else None,
            "num_false_rejections": int(num_false_rejections) if num_false_rejections is not None else None,
            "num_tests": int(len(hold_hold_t_stats)),
        },
        "tpr": {
            "value": float(tpr) if tpr is not None else None,
            "num_true_rejections": int(num_true_rejections) if num_true_rejections is not None else None,
            "num_tests": int(len(sus_hold_t_stats)),
        }
    }


def main():
    """Main function to calculate TPR/FPR for all dataset sizes"""
    DATASET_SIZES = [10, 20, 50, 100, 200, 300, 400, 500,]
    ALPHA = 0.001
    K_BASE = 400
    
    model_name = attack_args.model_name
    
    ROOT_DIR = os.path.join("exp_data", "ttest")
    
    print("="*60)
    print("Calculating TPR and FPR for Loop Results")
    print("="*60)
    print(f"Alpha: {ALPHA}")
    print(f"K_BASE: {K_BASE}")
    print(f"Dataset sizes: {DATASET_SIZES}")
    print("="*60)
    
    all_results = {}
    
    for size in DATASET_SIZES:
        ttest_dir = os.path.join(ROOT_DIR, str(size), "results")
        
        if not os.path.exists(ttest_dir):
            print(f"\nSize {size}: Directory not found - {ttest_dir}")
            continue
        
        print(f"\nProcessing size={size}...")
        print(f"  Directory: {ttest_dir}")
        
        result = calculate_tpr_fpr_for_size(ttest_dir, alpha=ALPHA, k_base=K_BASE)
        
        if result:
            all_results[size] = result
            
            tpr_str = f"{result['tpr']['value']:.4f}" if result['tpr']['value'] is not None else "N/A"
            fpr_str = f"{result['fpr']['value']:.4f}" if result['fpr']['value'] is not None else "N/A"
            
            print(f"  tau: {result['tau']:.4f}")
            print(f"  Base12 tests: {result['base12']['num_t_stats']}")
            print(f"  TPR: {tpr_str} ({result['tpr']['num_true_rejections']}/{result['tpr']['num_tests']})")
            print(f"  FPR: {fpr_str} ({result['fpr']['num_false_rejections']}/{result['fpr']['num_tests']})")
        else:
            print(f"  Failed to calculate TPR/FPR")
    
    summary_data = {
        "model_name": model_name,
        "alpha": ALPHA,
        "k_base": K_BASE,
        "dataset_sizes": DATASET_SIZES,
        "results": all_results
    }
    
    summary_file = os.path.join(ROOT_DIR, "loop_tpr_fpr_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)
    print(f"{'Dataset Size':<15} {'TPR':<12} {'FPR':<12} {'tau':<12} {'Base12 N':<12} {'SusHold N':<12} {'HoldHold N':<12}")
    print("-"*80)
    
    for size in DATASET_SIZES:
        if size in all_results:
            r = all_results[size]
            tpr_str = f"{r['tpr']['value']:.4f}" if r['tpr']['value'] is not None else "N/A"
            fpr_str = f"{r['fpr']['value']:.4f}" if r['fpr']['value'] is not None else "N/A"
            print(f"{size:<15} {tpr_str:<12} {fpr_str:<12} {r['tau']:<12.4f} {r['base12']['num_t_stats']:<12} {r['tpr']['num_tests']:<12} {r['fpr']['num_tests']:<12}")
    
    print("="*80)
    print(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    main()
