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
    print(f"{'='*60}")
    
    ds1_feats, ds2_feats = loader.load_dataset_pair_features(dataset1_name, dataset2_name, test_i)
    
    print(f"  DS1 features shape: {ds1_feats.shape}")
    print(f"  DS2 features shape: {ds2_feats.shape}")
    
    cdi_result_file_path = os.path.join(CDI_OUTPUT_DIR, f"cdi_test_{test_i}_results.json")
    
    print(f"Running CDI score computation...")
    members_scores, nonmembers_scores = cdi_process_data(ds1_feats, ds2_feats, test_i)
    
    members_scores_np = members_scores.cpu().numpy()
    nonmembers_scores_np = nonmembers_scores.cpu().numpy()
    
    p_value, is_correct_order, t_statistic = get_p_value(members_scores_np, nonmembers_scores_np, members_lower=False)
    
    print(f"CDI Result - p-value: {p_value:.6f}, is_correct_order: {is_correct_order}")
    
    result = {
        "test_i": test_i,
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        "p_value": float(p_value),
        "t_statistic": float(t_statistic),
        "is_correct_order": is_correct_order,
        "members_scores": members_scores_np.tolist(),
        "nonmembers_scores": nonmembers_scores_np.tolist()
    }
    
    with open(cdi_result_file_path, "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved to {cdi_result_file_path}")
    return result


if __name__ == "__main__":
    SPLIT_TIMES = 20  # Number of repeated tests
    
    dataset1_name = attack_args.used_datasets[0]
    dataset2_name = attack_args.used_datasets[1]
    
    CDI_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "cdi_baseline")
    os.makedirs(CDI_OUTPUT_DIR, exist_ok=True)
    
    print("Initializing Metadata & Feature Loader...")
    loader = create_loader()
    
    print(f"Starting CDI baseline on dataset pair: {dataset1_name} vs {dataset2_name}")
    all_results = []
    for test_i in range(SPLIT_TIMES):
        try:
            result = process_single_pair(
                        loader, 
                        test_i, 
                        dataset1_name, 
                        dataset2_name,
                        CDI_OUTPUT_DIR
                    )
            all_results.append(result)
        except FileNotFoundError as e:
                print(f"\nError: {e}")
     
    print(f"\nSummary Statistics:")
    if all_results:
        p_values = [result['p_value'] for result in all_results]
        correct_orders = [result['is_correct_order'] for result in all_results]
        print(f"  Min p-value: {np.min(p_values):.6f}")
        print(f"  Max p-value: {np.max(p_values):.6f}")
        print(f"  Mean p-value: {np.mean(p_values):.6f}")
        print(f"  Correct order rate: {np.mean(correct_orders):.2%}")
