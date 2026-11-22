import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import OUTPUT_ROOT_PATH, attack_args
import numpy as np
import json
import random


def sample_calibration_pairs(pair_names, k_base, seed=0):
    return random.Random(seed).sample(sorted(pair_names), k_base)


def load_t_stats_from_dir(ttest_dir, k_base=None):
    """
    Load t-statistics from a directory, returning both stats and dataset info
    
    Args:
        ttest_dir: Directory containing t-test results
        k_base: Number of base12 pairs to load (None for all)
    """
    all_t_stats = []
    dataset_info_list = []
    
    if not os.path.exists(ttest_dir):
        return np.array([]), []
    
    # Look for all pair_* subdirectories
    pair_dirs = [d for d in os.listdir(ttest_dir) if d.startswith("pair_") and os.path.isdir(os.path.join(ttest_dir, d))]
    pair_dirs.sort()
    
    # Separate base12 pairs (numeric pair_idx) from others
    base12_pair_dirs = []
    other_pair_dirs = []
    
    for pair_dir in pair_dirs:
        pair_idx = pair_dir.split("_")[1]
        if pair_idx.isdigit():
            base12_pair_dirs.append(pair_dir)
        else:
            other_pair_dirs.append(pair_dir)
    
    # Sample calibration pairs without replacement.
    if k_base is not None:
        base12_pair_dirs = sample_calibration_pairs(base12_pair_dirs, k_base)
    
    # Combine base12 and other pairs
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
        # calculate quantile thresholds
        lower = np.percentile(stats, 100 * trim_percent)
        upper = np.percentile(stats, 100 * (1 - trim_percent))
        mean_val = np.mean(stats)
    
        # replace outliers with mean
        mask = (stats < lower) | (stats > upper)
        cleaned_stats = np.where(mask, mean_val, stats)
    else:
        cleaned_stats = stats # use original stats if trim_percent is None
    # calculate empirical quantile threshold
    q = 100 * (1 - alpha)
    tau_quantile = np.percentile(cleaned_stats, q)
    
    return {
        "tau_quantile": float(tau_quantile),
        "alpha": alpha,
    }

def is_base12_pair(ds1, ds2):
    return {ds1, ds2} == {"D_base_1", "D_base_2"}


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


def main(k_base=None):
    if attack_args.experiment == "rq2_rq5":
        run_dir = os.path.join(OUTPUT_ROOT_PATH, "experiments", "shared_res", "RQ2_RQ5_calibration_pool", "intermediate", attack_args.model_name)
    else:
        run_dir = os.path.join(OUTPUT_ROOT_PATH, "experiments", "RQ3_calibration_performance", attack_args.model_name)
    DEFAULT_TTEST_DIR = os.path.join(run_dir, "ttest")
    # parameters
    DEFAULT_K_BASE = attack_args.k_base
    SPLIT_TIMES = attack_args.split_times
    ALPHA = 0.001 # significance level 
    
    # use provided k_base or default
    K_BASE = k_base if k_base is not None else DEFAULT_K_BASE
    
    # create output directory with k_base suffix if specified
    if k_base is None:
        RESULTS_OUTPUT_DIR = os.path.join(run_dir, "threshold_fpr")
    else:
        RESULTS_OUTPUT_DIR = os.path.join(run_dir, f"threshold_fpr_kbase{K_BASE}")
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("Calculating Threshold tau, TPR, and FPR")
    print("="*60)
    
    # Default to single directory mode
    print(f"\nReading t-test results from: {DEFAULT_TTEST_DIR}")
    print(f"Using K_BASE = {K_BASE} pairs of base12 data")
    all_t_stats, all_dataset_info = load_t_stats_from_dir(DEFAULT_TTEST_DIR, k_base=K_BASE)
    
    if len(all_t_stats) == 0:
        print("Error: No t-test results found")
        return
    
    # Separate into different categories
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
    
    # Print summary of loaded data
    print(f"\nLoaded data summary:")
    print(f"Total results: {len(all_t_stats)}")
    print(f"Base12 results: {len(base12_t_stats)}")
    if base12_info:
        print(f"Dataset pair: {base12_info['dataset1']} vs {base12_info['dataset2']}")
    print(f"Hold-Hold results: {len(hold_hold_t_stats)}")
    if hold_hold_info:
        print(f"Dataset pair: {hold_hold_info['dataset1']} vs {hold_hold_info['dataset2']}")
    print(f"Sus-Hold results: {len(sus_hold_t_stats)}")
    if sus_hold_info:
        print(f"Dataset pair: {sus_hold_info['dataset1']} vs {sus_hold_info['dataset2']}")
    
    # Check if we have base12 data
    if len(base12_t_stats) == 0:
        print("Error: No base12 data loaded")
        return
    
    # Calculate threshold tau (Empirical Quantile rule)
    print(f"\nCalculating threshold tau using empirical quantile (alpha={ALPHA})...")
    quantile_result = empirical_quantile_mean(base12_t_stats, alpha=ALPHA)
    tau = quantile_result["tau_quantile"]
    print(f"Threshold tau (Quantile) = {tau:.6f}")
    
    # Calculate FPR
    fpr = None
    num_false_rejections = None
    if len(hold_hold_t_stats) > 0:
        fpr, num_false_rejections = calculate_rejection_rate(hold_hold_t_stats, tau)
        print(f"\nFPR calculation on D_hold/D_hold:")
        print(f"Number of tests: {len(hold_hold_t_stats)}")
        print(f"Number of false rejections (t > tau): {num_false_rejections}")
        print(f"FPR: {fpr:.6f}")
    
    # Calculate TPR
    tpr = None
    num_true_rejections = None
    if len(sus_hold_t_stats) > 0:
        tpr, num_true_rejections = calculate_rejection_rate(sus_hold_t_stats, tau)
        print(f"\nTPR calculation on D_sus/D_hold:")
        print(f"Number of tests: {len(sus_hold_t_stats)}")      
        print(f"Number of true rejections (t > tau): {num_true_rejections}")
        print(f"TPR: {tpr:.6f}")
    
    
    # Save all results
    results_dict = {
        "parameters": {
            "K_BASE": int(K_BASE),
            "SPLIT_TIMES": int(SPLIT_TIMES),
            "alpha": float(ALPHA),
            "threshold_method": "empirical_quantile"
        },
        "base12_statistics": {
            "dataset1": base12_info['dataset1'] if base12_info else None,
            "dataset2": base12_info['dataset2'] if base12_info else None,
            "num_t_stats": int(len(base12_t_stats)),
            "min_t": float(np.min(base12_t_stats)),
            "max_t": float(np.max(base12_t_stats)),
            "mean_t": float(np.mean(base12_t_stats)),
            "std_t": float(np.std(base12_t_stats)),
            "tau": float(tau)
        },
        "hold_hold_statistics": {
            "dataset1": hold_hold_info['dataset1'] if hold_hold_info else None,
            "dataset2": hold_hold_info['dataset2'] if hold_hold_info else None,
            "num_t_stats": int(len(hold_hold_t_stats)),
            "min_t": float(np.min(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None,
            "max_t": float(np.max(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None,
            "mean_t": float(np.mean(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None,
            "std_t": float(np.std(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None,
            "num_false_rejections": int(num_false_rejections) if num_false_rejections is not None else None,
            "fpr": float(fpr) if fpr is not None else None
        },
        "sus_hold_statistics": {
            "dataset1": sus_hold_info['dataset1'] if sus_hold_info else None,
            "dataset2": sus_hold_info['dataset2'] if sus_hold_info else None,
            "num_t_stats": int(len(sus_hold_t_stats)),
            "min_t": float(np.min(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None,
            "max_t": float(np.max(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None,
            "mean_t": float(np.mean(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None,
            "std_t": float(np.std(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None,
            "num_true_rejections": int(num_true_rejections) if num_true_rejections is not None else None,
            "tpr": float(tpr) if tpr is not None else None
        },
    }
    
    output_file = os.path.join(RESULTS_OUTPUT_DIR, "threshold_fpr_results.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"  Threshold tau (Quantile, alpha={ALPHA}): {tau:.6f}")
    if fpr is not None:
        print(f"  FPR on D_hold/D_hold: {fpr:.6f}")
    if tpr is not None:
        print(f"  TPR on D_sus/D_hold: {tpr:.6f}")
    print("="*60)


if __name__ == "__main__":

    main()
