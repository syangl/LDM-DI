import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args, OUTPUT_ROOT_PATH
from exp_utils.dataset_process_util.meta_feature_manager import create_loader
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind
import json
import datetime
import time


class MIDataset:
    def __init__(
        self,
        member_data: torch.Tensor,
        nonmember_data: torch.Tensor,
    ):
        self.data = torch.concat([member_data, nonmember_data])
        self.label = torch.concat(
            [torch.ones(member_data.size(0)), torch.zeros(nonmember_data.size(0))]
        )

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

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

def get_datasets_clf(members: torch.Tensor, nonmembers: torch.Tensor) -> MIDataset:
    dataset = MIDataset(members, nonmembers)
    return dataset


def asserts(members_scores: np.ndarray, nonmembers_scores: np.ndarray) -> None:
    assert len(members_scores.shape) == 1
    assert len(nonmembers_scores.shape) == 1
    assert members_scores.shape[0] > 0
    assert nonmembers_scores.shape[0] > 0
    assert members_scores.shape == nonmembers_scores.shape


def get_p_value(members_scores: np.ndarray, nonmembers_scores: np.ndarray, members_lower: bool = False):
    asserts(members_scores, nonmembers_scores)
    
    result = ttest_ind(members_scores, nonmembers_scores, equal_var=False)
    p_value = result.pvalue
    t_statistic = result.statistic

    is_correct_order = (t_statistic < 0 and members_lower) or (
        t_statistic > 0 and not members_lower
    )

    return p_value, is_correct_order, t_statistic


def cdi_process_data(members: torch.Tensor, nonmembers: torch.Tensor, test_i: int):
    members_indices = np.arange(len(members))    
    kf = KFold(n_splits=5, shuffle=True, random_state=test_i)
    members_scores, nonmembers_scores = [], []

    for train_idx, test_idx in kf.split(members_indices):
        members_train = members[train_idx]
        members_test = members[test_idx]
        nonmembers_train = nonmembers[train_idx]
        nonmembers_test = nonmembers[test_idx]
        
        train_dataset = get_datasets_clf(members_train, nonmembers_train)
        test_dataset = get_datasets_clf(members_test, nonmembers_test)
        
        ss = StandardScaler()
        train_data = torch.from_numpy(ss.fit_transform(train_dataset.data))
        test_data = torch.from_numpy(ss.transform(test_dataset.data))
        
        clf = LogisticRegression(random_state=0, max_iter=1000, n_jobs=None, solver="liblinear")
        clf.fit(train_data, train_dataset.label)
        
        scores = torch.from_numpy(clf.predict_proba(test_data)[:, 1])
        
        members_scores.append(scores[: len(test_dataset.data) // 2])
        nonmembers_scores.append(scores[len(test_dataset.data) // 2 :])
        
    members_scores = torch.cat(members_scores)
    nonmembers_scores = torch.cat(nonmembers_scores)

    assert len(members_scores) == len(nonmembers_scores) == (len(test_dataset.data) + len(train_dataset.data)) // 2
    return members_scores, nonmembers_scores


def process_single_pair(loader, test_i, dataset1_name, dataset2_name, CDI_OUTPUT_DIR):
    print(f"\n{'='*60}")
    print(f"Processing Test {test_i}: {dataset1_name} vs {dataset2_name} (CDI baseline)")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # record start time
    test_start_time = time.time()
    
    ds1_feats, ds2_feats = loader.load_dataset_pair_features(dataset1_name, dataset2_name, test_i)
    
    print(f"  DS1 features shape: {ds1_feats.shape}")
    print(f"  DS2 features shape: {ds2_feats.shape}")
    
    if is_hold_hold_pair(dataset1_name, dataset2_name):
        os.makedirs(os.path.join(CDI_OUTPUT_DIR, "hold_hold"), exist_ok=True)
        cdi_result_file_path = os.path.join(CDI_OUTPUT_DIR, "hold_hold", f"cdi_test_{test_i}_results.json")
    elif is_sus_hold_pair(dataset1_name, dataset2_name):
        os.makedirs(os.path.join(CDI_OUTPUT_DIR, "sus_hold"), exist_ok=True)
        cdi_result_file_path = os.path.join(CDI_OUTPUT_DIR, "sus_hold", f"cdi_test_{test_i}_results.json")
    else:
        raise ValueError(f"Unknown dataset pair: {dataset1_name} vs {dataset2_name}")
    
    print(f"Running CDI score computation...")
    members_scores, nonmembers_scores = cdi_process_data(ds1_feats, ds2_feats, test_i)
    
    members_scores_np = members_scores.cpu().numpy()
    nonmembers_scores_np = nonmembers_scores.cpu().numpy()
    
    p_value, is_correct_order, t_statistic = get_p_value(members_scores_np, nonmembers_scores_np, members_lower=False)
    
    # calculate elapsed time
    test_elapsed = time.time() - test_start_time
    hours, rem = divmod(test_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"CDI Result - p-value: {p_value:.6f}, is_correct_order: {is_correct_order}")
    print(f"Test {test_i} elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    result = {
        "test_i": test_i,
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        "p_value": float(p_value),
        "t_statistic": float(t_statistic),
        "is_correct_order": bool(is_correct_order),
        "test_elapsed_seconds": test_elapsed,
        "members_scores": members_scores_np.tolist(),
        "nonmembers_scores": nonmembers_scores_np.tolist()
    }
    
    with open(cdi_result_file_path, "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved to {cdi_result_file_path}")
    return result, test_elapsed


if __name__ == "__main__":
    torch.manual_seed(attack_args.global_seed)
    SPLIT_TIMES = attack_args.split_times  # Number of repeated tests
    
    dataset1_name = attack_args.used_datasets[0]
    dataset2_name = attack_args.used_datasets[1]
    
    CDI_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "experiments", "RQ1_standard_di_vulnerability", "cdi_baseline", attack_args.model_name)
    os.makedirs(CDI_OUTPUT_DIR, exist_ok=True)
    
    # time tracking
    time_stats = {
        "model_name": attack_args.model_name,
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        "start_time": datetime.datetime.now().isoformat(),
        "tests": {},
        "total_time_seconds": 0
    }
    
    # record overall start time
    overall_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Starting CDI baseline on dataset pair: {dataset1_name} vs {dataset2_name}")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    print("\nInitializing Metadata & Feature Loader...")
    loader = create_loader()
    
    all_results = []
    for test_i in range(SPLIT_TIMES):
        try:
            result, test_elapsed = process_single_pair(
                        loader, 
                        test_i, 
                        dataset1_name, 
                        dataset2_name,
                        CDI_OUTPUT_DIR
                    )
            all_results.append(result)
            
            # Save test time
            time_stats["tests"][str(test_i)] = test_elapsed
            time_stats["total_time_seconds"] += test_elapsed
        except FileNotFoundError as e:
                raise
    
    # calculate overall elapsed time
    overall_elapsed = time.time() - overall_start_time
    time_stats["end_time"] = datetime.datetime.now().isoformat()
    time_stats["total_time_seconds"] = overall_elapsed
    
    # format time
    total_hours, total_rem = divmod(overall_elapsed, 3600)
    total_minutes, total_seconds = divmod(total_rem, 60)
    time_stats["total_time_formatted"] = f"{int(total_hours)}h {int(total_minutes)}m {total_seconds:.2f}s"
    
    measured_tests = [r['test_elapsed_seconds'] for r in all_results if 'test_elapsed_seconds' in r]
    if measured_tests:
        time_stats["mean_test_seconds"] = float(np.mean(measured_tests))

    # save time stats
    time_stats_file = os.path.join(CDI_OUTPUT_DIR, f"time_stats_{dataset1_name}_{dataset2_name}.json")
    with open(time_stats_file, "w", encoding='utf-8') as f:
        json.dump(time_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary Statistics:")
    if all_results:
        p_values = [result['p_value'] for result in all_results]
        correct_orders = [result['is_correct_order'] for result in all_results]
        print(f"  Min p-value: {np.min(p_values):.6f}")
        print(f"  Max p-value: {np.max(p_values):.6f}")
        print(f"  Mean p-value: {np.mean(p_values):.6f}")
        print(f"  Correct order rate: {np.mean(correct_orders):.2%}")
    
    # print final summary
    print(f"\n{'='*60}")
    print(f"Total elapsed time: {time_stats['total_time_formatted']}")
    print(f"Time stats saved to: {time_stats_file}")
    print(f"{'='*60}")
