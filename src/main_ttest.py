import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args, OUTPUT_ROOT_PATH
from exp_utils.dataset_process_util.meta_feature_manager import create_loader
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind
import json
import matplotlib.pyplot as plt
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
        # torch.nn.Sigmoid()
    )
    save_file = os.path.join(MODEL_OUTPUT_DIR, f"pair_{pair_idx}", f"model_{pair_idx}_{test_i}.pt")
    if os.path.exists(save_file):
        print(f"{save_file} already exists, skipping training.")
        model.load_state_dict(torch.load(save_file))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    all_features = torch.cat((sus_features, hold_out_features), dim=0)
    all_labels = torch.cat((sus_label, hold_out_label), dim=0)
    total_num = all_features.size(0)

    print(f"Start training model {pair_idx}_{test_i}...")
    epochs = 1000
    loss_list = []
    model.train()
    with tqdm(range(epochs), desc=f"Training model {pair_idx}_{test_i}", leave=False) as tqdm_bar:
        for epoch in range(epochs):
            indices = torch.randperm(total_num)
            tmp_features = all_features[indices]
            tmp_labels = all_labels[indices]

            optimizer.zero_grad()
            outputs = model(tmp_features)
            loss = criterion(outputs, tmp_labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            tqdm_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
            tqdm_bar.update(1)

    plt.figure()
    plt.title(f"Training Loss for Model {pair_idx}_{test_i}")
    plt.xlim(0, epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tick_params(axis='both', which='both', direction="in")
    plt.tight_layout()
    plt.plot(loss_list)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"pair_{pair_idx}", f"training_loss_{pair_idx}_{test_i}.png"))
    plt.close()

    torch.save(model.state_dict(), save_file)
    return model


def predict(model, features):
    model.eval()
    logits = model(features).detach()
    return torch.sigmoid(logits)


def t_test(B_sus, B_hold_out):
    t, _ = ttest_ind(B_sus, B_hold_out, equal_var=False, alternative='greater')
    return t.item()


def t_test_with_effect(B_sus, B_hold_out):
    _, p = ttest_ind(B_sus, B_hold_out, equal_var=False, alternative='greater')
    neg_log_p = -np.log(p) if p > 0 else -np.inf

    m1, m2 = np.mean(B_sus), np.mean(B_hold_out)
    s1, s2 = np.std(B_sus, ddof=1), np.std(B_hold_out, ddof=1)
    n1, n2 = len(B_sus), len(B_hold_out)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp if sp > 0 else 0

    return p.item(), neg_log_p.item(), d.item()


def process_single_pair(loader, pair_idx, dataset1_name, dataset2_name, MODEL_OUTPUT_DIR, TTest_OUTPUT_DIR, SPLIT_TIMES, SPLIT_DS_LEN):
    """
    Process a single dataset pair for t-test
    """
    ttest_file_path = os.path.join(TTest_OUTPUT_DIR, f"pair_{pair_idx}", f"ttest_pair_{pair_idx}_results.json")
    
    # check if results already exist
    if os.path.exists(ttest_file_path):
        print(f"\n{'='*60}")
        print(f"Results for Pair {pair_idx} already exist: {ttest_file_path}")
        print(f"{'='*60}")
        print("Loading existing results...")
        with open(ttest_file_path, "r", encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results")
        return results, 0
    
    print(f"\n{'='*60}")
    print(f"Processing Pair {pair_idx}: {dataset1_name} vs {dataset2_name}")
    print(f"{'='*60}")
    
    # record start time for this pair
    pair_start_time = time.time()
    print(f"Pair {pair_idx} start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # load features for this pair
    ds1_feats, ds2_feats = loader.load_dataset_pair_features(dataset1_name, dataset2_name, pair_idx)
    # if dataset_total_num is less than loaded features, randomly sample to the target size
    if attack_args.dataset_total_num < ds1_feats.shape[0]:
        target_size = attack_args.dataset_total_num
        # use fixed seed for reproducibility
        sampler = torch.Generator().manual_seed(0)
        sampled_indices = torch.randperm(ds1_feats.shape[0], generator=sampler)[:target_size]
        ds1_feats = ds1_feats[sampled_indices]
        ds2_feats = ds2_feats[sampled_indices]
        print(f"  Sampled {target_size} features for both datasets")

    
    print(f"  DS1 features shape: {ds1_feats.shape}")
    print(f"  DS2 features shape: {ds2_feats.shape}")
    
    total_ds_len = ds1_feats.shape[0]
    
    results = []
    
    for test_i in range(SPLIT_TIMES):
        preds_file_path = os.path.join(TTest_OUTPUT_DIR, f"pair_{pair_idx}", f"pair_{pair_idx}_preds_{test_i}.json")
        
        print(f"\n--- Start Testing Pair {pair_idx} - Test {test_i} ---")
        test_start_time = time.time()
        
        # generate exact permutation indices
        generator = torch.Generator().manual_seed(test_i)
        indices = torch.randperm(total_ds_len, generator=generator)
        A_indices = indices[:SPLIT_DS_LEN]
        B_indices = indices[SPLIT_DS_LEN:]
        
        # split A and B using ds1 as "sus" and ds2 as "hold_out"
        features_A_tuple = process_features(ds1_feats[A_indices], ds2_feats[A_indices])
        features_B_tuple = process_features(ds1_feats[B_indices], ds2_feats[B_indices])
        
        # train model on A
        model = train_model(
            features_A_tuple[0], 
            features_A_tuple[1],
            features_A_tuple[2], 
            features_A_tuple[3],
            pair_idx,
            test_i, 
            MODEL_OUTPUT_DIR
        )
        
        # predict on B
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
                "B_sus_labels": features_B_tuple[1].cpu().numpy().tolist(),
                "B_hold_out_labels": features_B_tuple[3].cpu().numpy().tolist()
            }, f, indent=4)
        
        # calculate Welch's t-statistic
        t_statistic = t_test(B_sus, B_hold_out)
        test_elapsed = time.time() - test_start_time
        print(f"Result Pair {pair_idx} Test {test_i}: t={t_statistic:.4f}, elapsed: {test_elapsed:.2f}s")
        
        results.append({
            "pair_idx": pair_idx,
            "dataset1": dataset1_name,
            "dataset2": dataset2_name,
            "test_i": test_i,
            "t_statistic": t_statistic,
            "test_elapsed_seconds": test_elapsed
        })
    
    # save all results for this pair
    with open(ttest_file_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    # calculate pair elapsed time
    pair_elapsed = time.time() - pair_start_time
    hours, rem = divmod(pair_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\n{'='*60}")
    print(f"Finished Pair {pair_idx}: {dataset1_name} vs {dataset2_name}")
    print(f"Pair {pair_idx} elapsed time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"{'='*60}")
    
    return results, pair_elapsed


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


if __name__ == "__main__":
    # parameters
    SPLIT_TIMES = attack_args.split_times  # Number of repeated tests
    SPLIT_DS_LEN = attack_args.dataset_total_num // 2
    K_BASE = attack_args.k_base  # Total pairs to process for base12
    
    # get dataset pair from attack_args
    dataset1_name = attack_args.used_datasets[0]
    dataset2_name = attack_args.used_datasets[1]
    
    # check if this is a base12 pair
    is_base12_pair = {dataset1_name, dataset2_name} == {"D_base_1", "D_base_2"}
    
    if attack_args.experiment == "rq2_rq5":
        run_dir = os.path.join(OUTPUT_ROOT_PATH, "experiments", "shared_res", "RQ2_RQ5_calibration_pool", "intermediate", attack_args.model_name)
    else:
        run_dir = os.path.join(OUTPUT_ROOT_PATH, "experiments", "RQ3_calibration_performance", attack_args.model_name)
    MODEL_OUTPUT_DIR = os.path.join(run_dir, "model")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    TTest_OUTPUT_DIR = os.path.join(run_dir, "ttest")
    os.makedirs(TTest_OUTPUT_DIR, exist_ok=True)
    
    # time tracking
    time_stats = {
        "model_name": attack_args.model_name,
        "dataset1": dataset1_name,
        "dataset2": dataset2_name,
        "start_time": datetime.datetime.now().isoformat(),
        "pairs": {},
        "cached_pairs": [],
        "total_time_seconds": 0
    }
    
    # record overall start time
    overall_start_time = time.time()
    print(f"\n{'='*60}")
    print(f"Starting t-test on dataset pair: {dataset1_name} vs {dataset2_name}")
    print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    # create the loader
    print("\nInitializing Metadata & Feature Loader...")
    loader = create_loader()
    
    all_pair_results = []
    
    if is_base12_pair:
        # process multiple pairs for base12
        for pair_idx in range(K_BASE):
            try:
                pair_results, pair_elapsed = process_single_pair(
                    loader, 
                    pair_idx + 1, 
                    dataset1_name, 
                    dataset2_name,
                    MODEL_OUTPUT_DIR, 
                    TTest_OUTPUT_DIR,
                    1, # ensure only K_BASE control the number of tests
                    SPLIT_DS_LEN
                )
                all_pair_results.extend(pair_results)
                
                if pair_elapsed == 0:
                    time_stats["cached_pairs"].append(str(pair_idx + 1))
                # Save pair time
                time_stats["pairs"][str(pair_idx + 1)] = pair_elapsed
                time_stats["total_time_seconds"] += pair_elapsed
            except FileNotFoundError as e:
                raise
    else:
        # process single pair for sus + hold-out dataset or hold + hold dataset
        try:
            # determine pair_idx based on dataset type
            if is_hold_hold_pair(dataset1_name, dataset2_name):
                pair_idx = "holdhold"
            elif is_sus_hold_pair(dataset1_name, dataset2_name):
                pair_idx = "sushold"
            else:
                raise ValueError(f"Unknown dataset pair type: {dataset1_name} vs {dataset2_name}")
            
            pair_results, pair_elapsed = process_single_pair(
                loader, 
                pair_idx, 
                dataset1_name, 
                dataset2_name,
                MODEL_OUTPUT_DIR, 
                TTest_OUTPUT_DIR,
                SPLIT_TIMES, 
                SPLIT_DS_LEN
            )
            all_pair_results.extend(pair_results)
            
            if pair_elapsed == 0:
                time_stats["cached_pairs"].append(str(pair_idx))
            # save pair time
            time_stats["pairs"][str(pair_idx)] = pair_elapsed
            time_stats["total_time_seconds"] += pair_elapsed
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
    
    measured_tests = [r['test_elapsed_seconds'] for r in all_pair_results if 'test_elapsed_seconds' in r]
    if measured_tests:
        time_stats["mean_test_seconds"] = float(np.mean(measured_tests))

    # save time stats
    time_stats_file = os.path.join(TTest_OUTPUT_DIR, f"time_stats_{dataset1_name}_{dataset2_name}.json")
    with open(time_stats_file, "w", encoding='utf-8') as f:
        json.dump(time_stats, f, indent=2, ensure_ascii=False)
    
    # save all combined results
    if all_pair_results:
        all_results_path = os.path.join(TTest_OUTPUT_DIR, f"ttest_pair_{dataset1_name}_{dataset2_name}_results.json")
        with open(all_results_path, "w", encoding='utf-8') as f:
            json.dump(all_pair_results, f, indent=4)
        print(f"\nSaved all results to {all_results_path}")
        
        # calculate summary statistics
        t_stats = [r['t_statistic'] for r in all_pair_results]
        print(f"\nSummary Statistics:")
        print(f"  Total tests: {len(t_stats)}")
        print(f"  Min t-statistic: {np.min(t_stats):.4f}")
        print(f"  Max t-statistic: {np.max(t_stats):.4f}")
    
    # print final summary
    print(f"\n{'='*60}")
    print(f"Total elapsed time: {time_stats['total_time_formatted']}")
    print(f"Time stats saved to: {time_stats_file}")
    print(f"{'='*60}")

