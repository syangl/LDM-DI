import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args, OUTPUT_ROOT_PATH
from exp_utils.dataset_process_util.metadata_manager import create_loader
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind, combine_pvalues
import json
import matplotlib.pyplot as plt
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


def train_model(sus_features, sus_label, hold_out_features, hold_out_label, test_i, MODEL_OUTPUT_DIR):
    model = torch.nn.Sequential(
        torch.nn.Linear(sus_features.shape[1], 1),
        torch.nn.Sigmoid()
    )
    save_file = os.path.join(MODEL_OUTPUT_DIR, f"model_{test_i}.pt")
    if os.path.exists(save_file):
        print(f"{save_file} already exists, skipping training.")
        model.load_state_dict(torch.load(save_file))
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    all_features = torch.cat((sus_features, hold_out_features), dim=0)
    all_labels = torch.cat((sus_label, hold_out_label), dim=0)
    total_num = all_features.size(0)

    print(f"Start training model {test_i}...")
    epochs = 1000
    loss_list = []
    model.train()
    with tqdm(range(epochs), desc=f"Training model {test_i}", leave=False) as tqdm_bar:
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
    plt.title(f"Training Loss for Model {test_i}")
    plt.xlim(0, epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tick_params(axis='both', which='both', direction="in")
    plt.tight_layout()
    plt.plot(loss_list)
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"training_loss_{test_i}.png"))
    plt.close()

    torch.save(model.state_dict(), save_file)
    return model


def predict(model, features):
    model.eval()
    return model(features).detach()


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


def process_single_pair(loader, pair_idx, MODEL_OUTPUT_DIR, TTest_OUTPUT_DIR, SPLIT_TIMES, SPLIT_DS_LEN):
    """
    Process a single dataset pair for t-test
    """
    print(f"\n{'='*60}")
    print(f"Processing Pair {pair_idx}")
    print(f"{'='*60}")
    
    # Load features for this pair
    ds1_feats, ds2_feats = loader.load_ds12_features(pair_idx)
    
    print(f"  DS1 features shape: {ds1_feats.shape}")
    print(f"  DS2 features shape: {ds2_feats.shape}")
    
    total_ds_len = ds1_feats.shape[0]
    
    ttest_file_path = os.path.join(TTest_OUTPUT_DIR, f"ttest_pair_{pair_idx}_results.json")
    
    results = []
    
    for test_i in range(SPLIT_TIMES):
        preds_file_path = os.path.join(TTest_OUTPUT_DIR, f"pair_{pair_idx}_preds_{test_i}.json")
        
        print(f"\n--- Start Testing Pair {pair_idx} - Test {test_i} ---")
        time_start = time.time()
        
        # Generate exact permutation indices
        generator = torch.Generator().manual_seed(test_i)
        indices = torch.randperm(total_ds_len, generator=generator)
        A_indices = indices[:SPLIT_DS_LEN]
        B_indices = indices[SPLIT_DS_LEN:]
        
        # Split A and B using ds1 as "sus" and ds2 as "hold_out"
        features_A = process_features(ds1_feats[A_indices], ds2_feats[A_indices])
        features_B = process_features(ds1_feats[B_indices], ds2_feats[B_indices])
        
        # Train model on A
        model = train_model(
            features_A[0], features_A[1],
            features_A[2], features_A[3],
            test_i, MODEL_OUTPUT_DIR
        )
        
        # Predict on B
        features_B = torch.cat((features_B[0], features_B[2]), dim=0)
        label_B = torch.cat((features_B[1], features_B[3]), dim=0)
        
        preds = predict(model, features_B)
        B_sus = preds[label_B == 1].cpu().numpy()
        B_hold_out = preds[label_B == 0].cpu().numpy()
        
        with open(preds_file_path, "w", encoding='utf-8') as f:
            json.dump({
                "B_sus_preds": B_sus.tolist(),
                "B_hold_out_preds": B_hold_out.tolist(),
                "B_sus_labels": features_B[1].cpu().numpy().tolist(),
                "B_hold_out_labels": features_B[3].cpu().numpy().tolist()
            }, f, indent=4)
        
        # Calculate Welch's t-statistic
        t_statistic = t_test(B_sus, B_hold_out)
        print(f"Result Pair {pair_idx} Test {test_i}: t={t_statistic:.4f}")
        print(f"Time for experiment: {time.time() - time_start:.2f} seconds")
        
        results.append({
            "pair_idx": pair_idx,
            "test_i": test_i,
            "t_statistic": t_statistic
        })
    
    # Save all results for this pair
    with open(ttest_file_path, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    return results


if __name__ == "__main__":
    # Parameters
    SPLIT_TIMES = 20  # Number of repeated tests
    SPLIT_DS_LEN = attack_args.dataset_total_num // 2
    NUM_PAIRS_TO_PROCESS = 100  # Total pairs to process
    
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "model")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    TTest_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "ttest")
    os.makedirs(TTest_OUTPUT_DIR, exist_ok=True)
    
    # Create the loader
    print("Initializing Metadata & Feature Loader...")
    loader = create_loader()
    
    all_pair_results = []
    
    print("Starting t-test on dataset pairs")
    # Process each pair
    for pair_idx in range(NUM_PAIRS_TO_PROCESS):
        try:
            pair_results = process_single_pair(
                loader, pair_idx, 
                MODEL_OUTPUT_DIR, TTest_OUTPUT_DIR,
                SPLIT_TIMES, SPLIT_DS_LEN
            )
            all_pair_results.extend(pair_results)
        except FileNotFoundError as e:
            print(f"\Error: Pair {pair_idx} not found, stopping.")
            # print(f"Error: {e}")
            break
    
    # Save all combined results
    if all_pair_results:
        all_results_path = os.path.join(TTest_OUTPUT_DIR, "all_ttest_results.json")
        with open(all_results_path, "w", encoding='utf-8') as f:
            json.dump(all_pair_results, f, indent=4)
        print(f"\nSaved all results to {all_results_path}")
        
        # Calculate summary statistics
        t_stats = [r['t_statistic'] for r in all_pair_results]
        print(f"\nSummary Statistics:")
        print(f"  Total tests: {len(t_stats)}")
        print(f"  Mean t-statistic: {np.mean(t_stats):.4f}")
        print(f"  Std t-statistic: {np.std(t_stats):.4f}")
        print(f"  Min t-statistic: {np.min(t_stats):.4f}")
        print(f"  Max t-statistic: {np.max(t_stats):.4f}")

