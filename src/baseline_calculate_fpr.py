import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import OUTPUT_ROOT_PATH
import numpy as np
import json


def load_results_from_dir(cdi_dir):
    """
    Load all CDI baseline results from a directory, returning both p-values and dataset info
    """
    all_p_values = []
    all_t_stats = []
    dataset_info_list = []
    
    if not os.path.exists(cdi_dir):
        return np.array([]), np.array([]), []
    
    # Look for hold_hold and sus_hold subdirectories
    subdirs = ["hold_hold", "sus_hold"]
    
    for subdir in subdirs:
        subdir_path = os.path.join(cdi_dir, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        result_files = [f for f in os.listdir(subdir_path) if f.startswith("cdi_test_") and f.endswith("_results.json")]
        result_files.sort()
        
        for result_file in result_files:
            result_file_path = os.path.join(subdir_path, result_file)
            
            with open(result_file_path, "r", encoding='utf-8') as f:
                result = json.load(f)
                all_p_values.append(result['p_value'])
                all_t_stats.append(result['t_statistic'])
                dataset_info = {
                    "dataset1": result.get('dataset1', 'unknown'),
                    "dataset2": result.get('dataset2', 'unknown'),
                    "p_value": result['p_value'],
                    "t_statistic": result['t_statistic'],
                    "subdir": subdir
                }
                dataset_info_list.append(dataset_info)
    
    return np.array(all_p_values), np.array(all_t_stats), dataset_info_list


def calculate_rejection_rate_by_pvalue(p_values, alpha):
    """
    Calculate rejection rate: proportion of p-values less than alpha (reject H0)
    """
    num_rejections = np.sum(p_values < alpha)
    rejection_rate = num_rejections / len(p_values) if len(p_values) > 0 else 0.0
    return rejection_rate, num_rejections


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


def main():
    DEFAULT_CDI_DIR = os.path.join(OUTPUT_ROOT_PATH, "cdi_baseline")
    # Parameters - standard significance levels
    ALPHA_LIST = [0.001]
    
    RESULTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "baseline_fpr")
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("Calculating Baseline FPR/TPR using p-value and alpha")
    print("="*60)
    
    # Read CDI baseline results
    print(f"\nReading CDI baseline results from: {DEFAULT_CDI_DIR}")
    all_p_values, _, all_dataset_info = load_results_from_dir(DEFAULT_CDI_DIR)
    
    if len(all_p_values) == 0:
        print("Error: No CDI baseline results found!")
        return
    
    # Separate into different categories
    hold_hold_p_values = []
    hold_hold_t_stats = []
    sus_hold_p_values = []
    sus_hold_t_stats = []
    hold_hold_info = None
    sus_hold_info = None
    
    for info in all_dataset_info:
        ds1, ds2 = info['dataset1'], info['dataset2']
        p = info['p_value']
        t = info['t_statistic']
        if is_hold_hold_pair(ds1, ds2):
            hold_hold_p_values.append(p)
            hold_hold_t_stats.append(t)
            if hold_hold_info is None:
                hold_hold_info = {"dataset1": ds1, "dataset2": ds2}
        elif is_sus_hold_pair(ds1, ds2):
            sus_hold_p_values.append(p)
            sus_hold_t_stats.append(t)
            if sus_hold_info is None:
                sus_hold_info = {"dataset1": ds1, "dataset2": ds2}
    
    hold_hold_p_values = np.array(hold_hold_p_values)
    hold_hold_t_stats = np.array(hold_hold_t_stats)
    sus_hold_p_values = np.array(sus_hold_p_values)
    sus_hold_t_stats = np.array(sus_hold_t_stats)
    
    # Print summary of loaded data
    print(f"\nLoaded data summary:")
    print(f"Total results: {len(all_p_values)}")
    print(f"Hold-Hold results: {len(hold_hold_p_values)}")
    if hold_hold_info:
        print(f"Dataset pair: {hold_hold_info['dataset1']} vs {hold_hold_info['dataset2']}")
    print(f"Sus-Hold results: {len(sus_hold_p_values)}")
    if sus_hold_info:
        print(f"Dataset pair: {sus_hold_info['dataset1']} vs {sus_hold_info['dataset2']}")
    
    # Calculate FPR and TPR for each alpha
    results_by_alpha = {}
    
    for alpha in ALPHA_LIST:
        print(f"\n{'='*60}")
        print(f"Results for alpha = {alpha}")
        print(f"{'='*60}")
        
        # Calculate FPR (false rejection rate on hold-hold pairs)
        fpr = None
        num_false_rejections = None
        if len(hold_hold_p_values) > 0:
            fpr, num_false_rejections = calculate_rejection_rate_by_pvalue(hold_hold_p_values, alpha)
            print(f"\nFPR calculation on D_hold/D_hold:")
            print(f"Number of tests: {len(hold_hold_p_values)}")
            print(f"Number of false rejections (p < alpha): {num_false_rejections}")
            print(f"FPR: {fpr:.6f}")
        
        # Calculate TPR (true rejection rate on sus-hold pairs)
        tpr = None
        num_true_rejections = None
        if len(sus_hold_p_values) > 0:
            tpr, num_true_rejections = calculate_rejection_rate_by_pvalue(sus_hold_p_values, alpha)
            print(f"\nTPR calculation on D_sus/D_hold:")
            print(f"Number of tests: {len(sus_hold_p_values)}")      
            print(f"Number of true rejections (p < alpha): {num_true_rejections}")
            print(f"TPR: {tpr:.6f}")
        
        results_by_alpha[alpha] = {
            "fpr": float(fpr) if fpr is not None else None,
            "num_false_rejections": int(num_false_rejections) if num_false_rejections is not None else None,
            "tpr": float(tpr) if tpr is not None else None,
            "num_true_rejections": int(num_true_rejections) if num_true_rejections is not None else None
        }
    
    # Save all results
    results_dict = {
        "parameters": {
            "alpha_list": [float(a) for a in ALPHA_LIST],
            "method": "p-value_alpha_test"
        },
        "hold_hold_statistics": {
            "dataset1": hold_hold_info['dataset1'] if hold_hold_info else None,
            "dataset2": hold_hold_info['dataset2'] if hold_hold_info else None,
            "num_p_values": int(len(hold_hold_p_values)),
            "min_p": float(np.min(hold_hold_p_values)) if len(hold_hold_p_values) > 0 else None,
            "max_p": float(np.max(hold_hold_p_values)) if len(hold_hold_p_values) > 0 else None,
            "mean_p": float(np.mean(hold_hold_p_values)) if len(hold_hold_p_values) > 0 else None,
            "std_p": float(np.std(hold_hold_p_values)) if len(hold_hold_p_values) > 0 else None,
            "min_t": float(np.min(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None,
            "max_t": float(np.max(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None,
            "mean_t": float(np.mean(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None,
            "std_t": float(np.std(hold_hold_t_stats)) if len(hold_hold_t_stats) > 0 else None
        },
        "sus_hold_statistics": {
            "dataset1": sus_hold_info['dataset1'] if sus_hold_info else None,
            "dataset2": sus_hold_info['dataset2'] if sus_hold_info else None,
            "num_p_values": int(len(sus_hold_p_values)),
            "min_p": float(np.min(sus_hold_p_values)) if len(sus_hold_p_values) > 0 else None,
            "max_p": float(np.max(sus_hold_p_values)) if len(sus_hold_p_values) > 0 else None,
            "mean_p": float(np.mean(sus_hold_p_values)) if len(sus_hold_p_values) > 0 else None,
            "std_p": float(np.std(sus_hold_p_values)) if len(sus_hold_p_values) > 0 else None,
            "min_t": float(np.min(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None,
            "max_t": float(np.max(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None,
            "mean_t": float(np.mean(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None,
            "std_t": float(np.std(sus_hold_t_stats)) if len(sus_hold_t_stats) > 0 else None
        },
        "results_by_alpha": results_by_alpha
    }
    
    output_file = os.path.join(RESULTS_OUTPUT_DIR, "baseline_fpr_results.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for alpha in ALPHA_LIST:
        result = results_by_alpha[alpha]
        print(f"\nalpha = {alpha}:")
        if result['fpr'] is not None:
            print(f"  FPR on D_hold/D_hold: {result['fpr']:.6f}")
        if result['tpr'] is not None:
            print(f"  TPR on D_sus/D_hold: {result['tpr']:.6f}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
