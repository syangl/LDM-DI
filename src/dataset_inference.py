import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args, TRAIN_OUTPUT_DIR, dataset_config_1, HOLD_OUT_OUTPUT_DIR, dataset_config_2, PIPE_TYPE, OUTPUT_ROOT_PATH
from exp_utils.load_util.load_data import create_dataset_dataloader, get_img_dataLoader
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select

from torch.utils.data import random_split
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind, combine_pvalues
import json
import matplotlib.pyplot as plt
import time


def clear_memory(*args):
    for arg in args:
        del arg
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def split_dataset(tokenizer, split_time, split="A"):
    # load dataset
    _, suspect_ds = create_dataset_dataloader(
        output_dir=TRAIN_OUTPUT_DIR,
        dataset_config=dataset_config_1,
        tokenizer=tokenizer,
        img_resolution=attack_args.generate_resolution,
        need_save=attack_args.is_visualize,
    )
    _, hold_out_ds = create_dataset_dataloader(
        output_dir=HOLD_OUT_OUTPUT_DIR,
        dataset_config=dataset_config_2,
        tokenizer=tokenizer,
        img_resolution=attack_args.generate_resolution,
        need_save=attack_args.is_visualize,
    )

    generator = torch.Generator().manual_seed(split_time)
    # split dataset
    sus_A_ds, sus_B_ds = random_split(suspect_ds, [SPLIT_DS_LEN, SPLIT_DS_LEN], generator=generator)
    hold_out_A_ds, hold_out_B_ds = random_split(hold_out_ds, [SPLIT_DS_LEN, SPLIT_DS_LEN], generator=generator)

    if split == 'A':
        return sus_A_ds, hold_out_A_ds
    else:
        return sus_B_ds, hold_out_B_ds


def feature_extract(pipe, sus_ds, hold_out_ds, test_i, split):
    def exec_mia(method, choice=''):
        if method == "structuralmi":
            print("Executing StructuralMI...")
            sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"structuralmi_{choice}_member_scores_{split}_{test_i}.json")
            hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"structuralmi_{choice}_nonmember_scores_{split}_{test_i}.json")
            if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
                # import
                src_path = os.path.join(current_dir, "exp_utils/mi_util/structuralmi_src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from exp_utils.mi_util.structuralmi_src.structuralmi import execute_structuralmi_attack
                # directory
                sus_scores_dir = os.path.join(TRAIN_OUTPUT_DIR, method)
                hold_out_scores_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, method)
                # create dataloaders
                sus_dataloader = get_img_dataLoader(sus_ds)
                hold_out_dataloader = get_img_dataLoader(hold_out_ds)
                # execute mia
                sus_scores = execute_structuralmi_attack(dataloader=sus_dataloader, pipe=pipe, choice=choice)
                hold_out_scores = execute_structuralmi_attack(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
                # save scores
                os.makedirs(sus_scores_dir, exist_ok=True)
                with open(sus_scores_file, "w", encoding='utf-8') as f:
                    json.dump(sus_scores, f)
                os.makedirs(hold_out_scores_dir, exist_ok=True)
                with open(hold_out_scores_file, "w", encoding='utf-8') as f:
                    json.dump(hold_out_scores, f)
            else:
                print(f"{sus_scores_file} & {hold_out_scores_file} already exists, skipping inference.")
                # load scores
                with open(sus_scores_file, "r", encoding='utf-8') as f:
                    sus_scores = json.load(f)
                with open(hold_out_scores_file, "r", encoding='utf-8') as f:
                    hold_out_scores = json.load(f)
            print(f"Member scores mean: {np.mean(sus_scores)}")
            print(f"Non-member scores mean: {np.mean(hold_out_scores)}")
            print()

        elif method == "secmi":
            print("Executing SecMI...")
            sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"secmi_member_scores_{split}_{test_i}.json")
            hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"secmi_nonmember_scores_{split}_{test_i}.json")
            if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
                # import
                src_path = os.path.join(current_dir, "exp_utils/mi_util/secmi_src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from exp_utils.mi_util.secmi_src.secmi import get_reverse_denoise_results
                # directory
                sus_scores_dir = os.path.join(TRAIN_OUTPUT_DIR, method)
                hold_out_scores_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, method)
                # create dataloaders
                sus_dataloader = get_img_dataLoader(sus_ds)
                hold_out_dataloader = get_img_dataLoader(hold_out_ds)
                # excute mia
                sus_scores = get_reverse_denoise_results(
                    pipe=pipe,
                    dataloader=sus_dataloader).cpu().numpy().tolist()
                # let member scores be larger
                sus_scores = [-score for score in sus_scores]
                hold_out_scores = get_reverse_denoise_results(
                    pipe=pipe,
                    dataloader=hold_out_dataloader).cpu().numpy().tolist()
                hold_out_scores = [-score for score in hold_out_scores]
                # save scores
                os.makedirs(sus_scores_dir, exist_ok=True)
                with open(sus_scores_file, "w", encoding='utf-8') as f:
                    json.dump(sus_scores, f)
                os.makedirs(hold_out_scores_dir, exist_ok=True)
                with open(hold_out_scores_file, "w", encoding='utf-8') as f:
                    json.dump(hold_out_scores, f)
            else:
                print(f"{sus_scores_file} & {hold_out_scores_file} already exists, skipping inference.")
                # load scores
                with open(sus_scores_file, "r", encoding='utf-8') as f:
                    sus_scores = json.load(f)
                with open(hold_out_scores_file, "r", encoding='utf-8') as f:
                    hold_out_scores = json.load(f)
            print(f"Member scores mean: {np.mean(sus_scores)}")
            print(f"Non-member scores mean: {np.mean(hold_out_scores)}")
            print()

        elif method == "naivelossmi":
            print("Executing NaiveLossMI...")
            sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"naivelossmi_member_scores_{split}_{test_i}.json")
            hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"naivelossmi_nonmember_scores_{split}_{test_i}.json")
            if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
                # import
                src_path = os.path.join(current_dir, "exp_utils/mi_util/naivelossmi_src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from exp_utils.mi_util.naivelossmi_src.naivelossmi import execute_naivelossmi_attack
                # directory
                sus_scores_dir = os.path.join(TRAIN_OUTPUT_DIR, method)
                hold_out_scores_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, method)
                # create dataloaders
                sus_dataloader = get_img_dataLoader(sus_ds)
                hold_out_dataloader = get_img_dataLoader(hold_out_ds)
                # execute mia
                sus_scores = execute_naivelossmi_attack(dataloader=sus_dataloader, pipe=pipe)
                hold_out_scores = execute_naivelossmi_attack(dataloader=hold_out_dataloader, pipe=pipe)
                # save scores
                os.makedirs(sus_scores_dir, exist_ok=True)
                with open(sus_scores_file, "w", encoding='utf-8') as f:
                    json.dump(sus_scores, f)
                os.makedirs(hold_out_scores_dir, exist_ok=True)
                with open(hold_out_scores_file, "w", encoding='utf-8') as f:
                    json.dump(hold_out_scores, f)
            else:
                print(f"{sus_scores_file} & {hold_out_scores_file} already exists, skipping inference.")
                # load scores
                with open(sus_scores_file, "r", encoding='utf-8') as f:
                    sus_scores = json.load(f)
                with open(hold_out_scores_file, "r", encoding='utf-8') as f:
                    hold_out_scores = json.load(f)
            print(f"Member scores mean: {np.mean(sus_scores)}")
            print(f"Non-member scores mean: {np.mean(hold_out_scores)}")
            print()

        elif method == "piami":
            print("Executing PiAMI...")
            sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"piami_member_scores_{split}_{test_i}.json")
            hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"piami_nonmember_scores_{split}_{test_i}.json")
            if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
                # import
                src_path = os.path.join(current_dir, "exp_utils/mi_util/piami_src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from exp_utils.mi_util.piami_src.piami import execute_piami_attack
                # directory
                sus_scores_dir = os.path.join(TRAIN_OUTPUT_DIR, method)
                hold_out_scores_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, method)
                # create dataloaders
                sus_dataloader = get_img_dataLoader(sus_ds)
                hold_out_dataloader = get_img_dataLoader(hold_out_ds)
                # execute mia
                sus_scores = execute_piami_attack(dataloader=sus_dataloader, pipe=pipe)
                hold_out_scores = execute_piami_attack(dataloader=hold_out_dataloader, pipe=pipe)
                # save scores
                os.makedirs(sus_scores_dir, exist_ok=True)
                with open(sus_scores_file, "w", encoding='utf-8') as f:
                    json.dump(sus_scores, f)
                os.makedirs(hold_out_scores_dir, exist_ok=True)
                with open(hold_out_scores_file, "w", encoding='utf-8') as f:
                    json.dump(hold_out_scores, f)
            else:
                print(f"{sus_scores_file} & {hold_out_scores_file} already exists, skipping inference.")
                # load scores
                with open(sus_scores_file, "r", encoding='utf-8') as f:
                    sus_scores = json.load(f)
                with open(hold_out_scores_file, "r", encoding='utf-8') as f:
                    hold_out_scores = json.load(f)
            print(f"Member scores mean: {np.mean(sus_scores)}")
            print(f"Non-member scores mean: {np.mean(hold_out_scores)}")
            print()

        elif method == "rediffusemi":
            print("Executing RediffuseMI...")
            sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"rediffusemi_{choice}_member_scores_{split}_{test_i}.json")
            hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"rediffusemi_{choice}_nonmember_scores_{split}_{test_i}.json")
            if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
                # import
                src_path = os.path.join(current_dir, "exp_utils/mi_util/rediffusemi_src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from exp_utils.mi_util.rediffusemi_src.rediffusemi import execute_rediffusemi_attack
                # directory
                sus_scores_dir = os.path.join(TRAIN_OUTPUT_DIR, method)
                hold_out_scores_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, method)
                # create dataloaders
                sus_dataloader = get_img_dataLoader(sus_ds)
                hold_out_dataloader = get_img_dataLoader(hold_out_ds)
                # execute mia
                sus_scores = execute_rediffusemi_attack(dataloader=sus_dataloader, pipe=pipe, choice=choice)
                hold_out_scores = execute_rediffusemi_attack(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
                # save scores
                os.makedirs(sus_scores_dir, exist_ok=True)
                with open(sus_scores_file, "w", encoding='utf-8') as f:
                    json.dump(sus_scores, f)
                os.makedirs(hold_out_scores_dir, exist_ok=True)
                with open(hold_out_scores_file, "w", encoding='utf-8') as f:
                    json.dump(hold_out_scores, f)
            else:
                print(f"{sus_scores_file} & {hold_out_scores_file} already exists, skipping inference.")
                # load scores
                with open(sus_scores_file, "r", encoding='utf-8') as f:
                    sus_scores = json.load(f)
                with open(hold_out_scores_file, "r", encoding='utf-8') as f:
                    hold_out_scores = json.load(f)
            print(f"Member scores mean: {np.mean(sus_scores)}")
            print(f"Non-member scores mean: {np.mean(hold_out_scores)}")
            print()

        elif method == "hintmi":
            print("Executing HintMI...")
            sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"hintmi_{choice}_member_scores_{split}_{test_i}.json")
            hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"hintmi_{choice}_nonmember_scores_{split}_{test_i}.json")
            if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
                # import
                src_path = os.path.join(current_dir, "exp_utils/mi_util/ours_hintmi_src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from exp_utils.mi_util.ours_hintmi_src.ours_hintmi import execute_hintmi
                # directory
                sus_scores_dir = os.path.join(TRAIN_OUTPUT_DIR, method)
                os.makedirs(sus_scores_dir, exist_ok=True)
                hold_out_scores_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, method)
                os.makedirs(hold_out_scores_dir, exist_ok=True)
                # create dataloaders
                sus_dataloader = get_img_dataLoader(sus_ds)
                hold_out_dataloader = get_img_dataLoader(hold_out_ds)
                # execute mia
                sus_scores = execute_hintmi(dataloader=sus_dataloader, pipe=pipe, choice=choice)
                hold_out_scores = execute_hintmi(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
                # save scores
                with open(sus_scores_file, "w", encoding='utf-8') as f:
                    json.dump(sus_scores, f)
                with open(hold_out_scores_file, "w", encoding='utf-8') as f:
                    json.dump(hold_out_scores, f)
            else:
                print(f"{sus_scores_file} & {hold_out_scores_file} already exists, skipping inference.")
                # load scores
                with open(sus_scores_file, "r", encoding='utf-8') as f:
                    sus_scores = json.load(f)
                with open(hold_out_scores_file, "r", encoding='utf-8') as f:
                    hold_out_scores = json.load(f)
            print(f"Member scores mean: {np.mean(sus_scores)}")
            print(f"Non-member scores mean: {np.mean(hold_out_scores)}")
            print()

        elif method == "semansim":
            print("Executing Prompt Sensitivity...")
            sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"semansim_member_scores_{split}_{test_i}.json")
            hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"semansim_nonmember_scores_{split}_{test_i}.json")
            if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
                # import
                src_path = os.path.join(current_dir, "exp_utils/mi_util/ours_semansim_src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from exp_utils.mi_util.ours_semansim_src.ours_semansim import execute_semansim
                # directory
                sus_scores_dir = os.path.join(TRAIN_OUTPUT_DIR, method)
                os.makedirs(sus_scores_dir, exist_ok=True)
                hold_out_scores_dir = os.path.join(HOLD_OUT_OUTPUT_DIR, method)
                os.makedirs(hold_out_scores_dir, exist_ok=True)
                # create dataloaders
                sus_dataloader = get_img_dataLoader(sus_ds)
                hold_out_dataloader = get_img_dataLoader(hold_out_ds)
                # execute mia
                sus_scores = execute_semansim(dataloader=sus_dataloader, pipe=pipe, choice=choice)
                hold_out_scores = execute_semansim(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
                # save scores
                with open(sus_scores_file, "w", encoding='utf-8') as f:
                    json.dump(sus_scores, f)
                with open(hold_out_scores_file, "w", encoding='utf-8') as f:
                    json.dump(hold_out_scores, f)
            else:
                print(f"{sus_scores_file} & {hold_out_scores_file} already exists, skipping inference.")
                # load scores
                with open(sus_scores_file, "r", encoding='utf-8') as f:
                    sus_scores = json.load(f)
                with open(hold_out_scores_file, "r", encoding='utf-8') as f:
                    hold_out_scores = json.load(f)
            print(f"Member scores mean: {np.mean(sus_scores)}")
            print(f"Non-member scores mean: {np.mean(hold_out_scores)}")
            print()

        else:
            raise ValueError(f"Unknown MIA method {method}")

        return sus_scores, hold_out_scores

    def feature_norm(features):
        features_mean = features.mean(dim=0, keepdim=True)
        features_std = features.std(dim=0, keepdim=True)
        return (features - features_mean) / (features_std + 1e-8)

    def handle_outliers(features, ratio=0.025):
        """
        Replace the top / bottom ratio part of each column feature with the column mean.
        """
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


    def convert2tensor32(mia_scores):
        if hasattr(mia_scores, 'cpu'):
            mia_scores = mia_scores.cpu().float()
        else:
            mia_scores = torch.from_numpy(np.array(mia_scores, dtype=np.float32))
        return mia_scores

    # get dataset features
    # 1. Current MIA methods
    mia_secmi_sus_scores, mia_secmi_hold_out_scores = exec_mia(method="secmi")
    mia_secmi_sus_scores = convert2tensor32(mia_secmi_sus_scores)
    mia_secmi_hold_out_scores = convert2tensor32(mia_secmi_hold_out_scores)

    mia_naivelossmi_sus_scores, mia_naivelossmi_hold_out_scores = exec_mia(method="naivelossmi")
    mia_naivelossmi_sus_scores = convert2tensor32(mia_naivelossmi_sus_scores)
    mia_naivelossmi_hold_out_scores = convert2tensor32(mia_naivelossmi_hold_out_scores)

    mia_piami_sus_scores, mia_piami_hold_out_scores = exec_mia(method="piami")
    mia_piami_sus_scores = convert2tensor32(mia_piami_sus_scores)
    mia_piami_hold_out_scores = convert2tensor32(mia_piami_hold_out_scores)

    mia_structuralmi_sus_scores, mia_structuralmi_hold_out_scores = exec_mia(method="structuralmi", choice="ssim")
    mia_structuralmi_sus_scores = convert2tensor32(mia_structuralmi_sus_scores)
    mia_structuralmi_hold_out_scores = convert2tensor32(mia_structuralmi_hold_out_scores)

    mia_rediffusemi_sus_scores, mia_rediffusemi_hold_out_scores = exec_mia(method="rediffusemi", choice="ssim")
    mia_rediffusemi_sus_scores = convert2tensor32(mia_rediffusemi_sus_scores)
    mia_rediffusemi_hold_out_scores = convert2tensor32(mia_rediffusemi_hold_out_scores)

    # 2. Our memory feature methods
    # 2.1 HintMI
    mia_hintmi_sus_scores, mia_hintmi_hold_out_scores = exec_mia(method="hintmi", choice="ssim")
    mia_hintmi_sus_scores = convert2tensor32(mia_hintmi_sus_scores)
    mia_hintmi_hold_out_scores = convert2tensor32(mia_hintmi_hold_out_scores)

    # 2.2 crop ssim similarity
    mia_ssim_center_sus_scores, mia_ssim_center_hold_out_scores = exec_mia(method="hintmi", choice="ssim_center")
    mia_ssim_center_sus_scores = convert2tensor32(mia_ssim_center_sus_scores)
    mia_ssim_center_hold_out_scores = convert2tensor32(mia_ssim_center_hold_out_scores)

    # 2.3 entropy feature
    mia_ssim_entropy_sus_scores, mia_ssim_entropy_hold_out_scores = exec_mia(method="hintmi", choice="ssim_entropy")
    mia_ssim_entropy_sus_scores = convert2tensor32(mia_ssim_entropy_sus_scores)
    mia_ssim_entropy_hold_out_scores = convert2tensor32(mia_ssim_entropy_hold_out_scores)

    # # 2.4 prompt sensitivity
    mia_semansim_sus_scores, mia_semansim_hold_out_scores = exec_mia(method="semansim", choice="cos_sim")
    mia_semansim_sus_scores = convert2tensor32(mia_semansim_sus_scores)
    mia_semansim_hold_out_scores = convert2tensor32(mia_semansim_hold_out_scores)


    # aggregate features
    sus_features = torch.cat((
        mia_secmi_sus_scores.unsqueeze(1),
        mia_naivelossmi_sus_scores.unsqueeze(1),
        mia_piami_sus_scores.unsqueeze(1),
        mia_structuralmi_sus_scores.unsqueeze(1),
        mia_rediffusemi_sus_scores.unsqueeze(1),
        mia_hintmi_sus_scores.unsqueeze(1),
        mia_ssim_center_sus_scores.unsqueeze(1),
        mia_ssim_entropy_sus_scores.unsqueeze(1),
        mia_semansim_sus_scores.unsqueeze(1),
        ), dim=1)
    hold_out_features = torch.cat((
        mia_secmi_hold_out_scores.unsqueeze(1),
        mia_naivelossmi_hold_out_scores.unsqueeze(1),
        mia_piami_hold_out_scores.unsqueeze(1),
        mia_structuralmi_hold_out_scores.unsqueeze(1),
        mia_rediffusemi_hold_out_scores.unsqueeze(1),
        mia_hintmi_hold_out_scores.unsqueeze(1),
        mia_ssim_center_hold_out_scores.unsqueeze(1),
        mia_ssim_entropy_hold_out_scores.unsqueeze(1),
        mia_semansim_hold_out_scores.unsqueeze(1),
        ), dim=1)

    # normalize
    all_features = torch.cat((sus_features, hold_out_features), dim=0)
    all_features_norm = feature_norm(all_features)
    sus_features_norm = all_features_norm[:sus_features.shape[0], :]
    hold_out_features_norm = all_features_norm[sus_features.shape[0]:, :]
    # handle outliers
    sus_features_norm = handle_outliers(sus_features_norm)
    hold_out_features_norm = handle_outliers(hold_out_features_norm)

    # label
    sus_label = torch.ones(sus_features_norm.shape[0]).unsqueeze(1)
    hold_out_label = torch.zeros(hold_out_features_norm.shape[0]).unsqueeze(1)

    return sus_features_norm, sus_label, hold_out_features_norm, hold_out_label



def train_model(sus_features, sus_label, hold_out_features, hold_out_label, test_i):
    def create_model(input_feature_dim):
        model = torch.nn.Sequential(
            torch.nn.Linear(input_feature_dim, 1),
            torch.nn.Sigmoid())
        return model
    # create model
    model = create_model(input_feature_dim=sus_features.shape[1])
    save_file = os.path.join(MODEL_OUTPUT_DIR, f"model_{test_i}.pt")
    if os.path.exists(save_file):
        print(f"{save_file} already exists, skipping training.")
        model.load_state_dict(torch.load(save_file))
        return model
    # create optimizer & criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # concatenate suspect and hold out data
    all_features = torch.cat((sus_features, hold_out_features), dim=0)
    all_labels = torch.cat((sus_label, hold_out_label), dim=0)
    total_num = all_features.size(0)

    # training
    print("Start training model...")
    epochs = 1000
    loss_list = []
    model.train()
    with tqdm(range(epochs), desc=f"Training model {test_i}") as tqdm_bar:
        for epoch in range(epochs):
            # shuffle data
            indices = torch.randperm(total_num)
            tmp_features = all_features[indices]
            tmp_labels = all_labels[indices]
            # train
            optimizer.zero_grad()
            outputs = model(tmp_features)
            loss = criterion(outputs, tmp_labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            tqdm_bar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
            tqdm_bar.update(1)

    # save training loss
    plt.figure()
    plt.title(f"Training Loss for Model {test_i}")
    plt.xlim(0, epochs)
    plt.xlabel("Epoch")
    # plt.ylim(0, 1.0)
    plt.ylabel("Loss")
    plt.tick_params(axis='both', which='both', direction="in")
    plt.tight_layout()
    plt.plot(loss_list)
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, f"training_loss_{test_i}.png"))
    plt.close()

    # save model
    print(f"Saving model {test_i}...")
    print()
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_DIR, f"model_{test_i}.pt"))
    return model


def predict(model, features):
    model.eval()
    outputs = model(features).detach()
    return outputs



def t_test(B_sus, B_hold_out):
    _, p = ttest_ind(B_sus, B_hold_out, equal_var=False, alternative='greater')
    return p


def t_test_with_effect(B_sus, B_hold_out):
    # Welch's ttest
    _, p = ttest_ind(B_sus, B_hold_out, equal_var=False, alternative='greater')

    # log_p
    neg_log_p = -np.log(p) if p > 0 else -np.inf

    # Cohen's d
    m1, m2 = np.mean(B_sus), np.mean(B_hold_out)
    s1, s2 = np.std(B_sus, ddof=1), np.std(B_hold_out, ddof=1)
    n1, n2 = len(B_sus), len(B_hold_out)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp

    return p.item(), neg_log_p.item(), d.item()



def aggregate_p_values(TTest_OUTPUT_DIR):
    p_values = []
    d_values = []
    n_values = []

    for test_i in range(SPLIT_TIMES):
        with open(os.path.join(TTest_OUTPUT_DIR, f"p_values_{test_i}.json"), "r") as f:
            results = json.load(f)
            p_values.append(results.get('p_value', 1.0))
            d_values.append(results['effect_size'])
            n_values.append(2 * SPLIT_DS_LEN)

    # Fisher's method for combining p-values
    _, p_combined = combine_pvalues(p_values, method='fisher')

    # log_p
    combined_log_p = np.log(p_combined) if p_combined > 0 else -np.inf

    # weighted average Cohen's d
    d_values = np.array(d_values, dtype=float)
    n_values = np.array(n_values, dtype=float)
    d_combined = np.sum(d_values * n_values) / np.sum(n_values)

    return p_combined, combined_log_p, d_combined



if __name__ == "__main__":
    # parameters
    SPLIT_TIMES = 5
    SPLIT_DS_LEN = attack_args.dataset_total_num // 2
    MODEL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "model")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    TTest_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "ttest")
    os.makedirs(TTest_OUTPUT_DIR, exist_ok=True)

    # Check which tests have existing predictions (Fast Path)
    # If preds_{i}.json exists, we don't need to load the dataset or the heavy diffusion pipeline
    preds_exist_flags = [os.path.exists(os.path.join(TTest_OUTPUT_DIR, f"preds_{i}.json")) for i in range(SPLIT_TIMES)]

    # Only load the heavy pipeline if we are missing predictions for at least one split
    if not all(preds_exist_flags):
        print("Some prediction files are missing. Loading pipeline for inference...")
        pipe = txt2img_pipeline_select(base_model_id=attack_args.base_model_id, PIPE_TYPE=PIPE_TYPE)
    else:
        print("All prediction files detected. Skipping pipeline loading to speed up p-value recalculation.")
        pipe = None

    for test_i in range(SPLIT_TIMES):
        preds_file_path = os.path.join(TTest_OUTPUT_DIR, f"preds_{test_i}.json")
        p_values_file_path = os.path.join(TTest_OUTPUT_DIR, f"p_values_{test_i}.json")

        # variables to hold the predictions
        B_sus = None
        B_hold_out = None
        if os.path.exists(preds_file_path):
            print(f"Found existing predictions for test {test_i}. Loading directly...")
            try:
                with open(preds_file_path, "r", encoding='utf-8') as f:
                    preds_data = json.load(f)
                    B_sus = np.array(preds_data["B_sus_preds"])
                    B_hold_out = np.array(preds_data["B_hold_out_preds"])
            except Exception as e:
                print(f"Error when loading {preds_file_path}:{e}. Excute experiment again.")
                B_sus = None 

        if B_sus is None:
            print(f"Start testing {test_i}...")
            print()
            print("Start A training...")
            print()
            # time start
            time_start = time.time()

            # split dataset
            sus_A_ds, hold_out_A_ds = split_dataset(pipe.tokenizer, test_i, "A")

            # feature extract
            sus_features_A, sus_label_A, hold_out_features_A, hold_out_label_A = feature_extract(
                pipe, sus_A_ds, hold_out_A_ds, test_i, "A")

            # release memory
            clear_memory(sus_A_ds, hold_out_A_ds)

            # train model
            model = train_model(sus_features_A, sus_label_A, hold_out_features_A, hold_out_label_A, test_i)

            # release memory
            clear_memory(sus_features_A, sus_label_A, hold_out_features_A, hold_out_label_A)

            # load split B
            print("Start B testing...")
            print()
            # load split B dataset
            sus_B_ds, hold_out_B_ds = split_dataset(pipe.tokenizer, test_i, "B")
            sus_features_B, sus_label_B, hold_out_features_B, hold_out_label_B = feature_extract(
                pipe, sus_B_ds, hold_out_B_ds, test_i, "B")
            features = torch.cat((sus_features_B, hold_out_features_B), dim=0)
            label = torch.cat((sus_label_B, hold_out_label_B), dim=0)
            # predict
            preds = predict(model, features)
            B_sus = preds[label == 1].cpu().numpy()
            B_hold_out = preds[label == 0].cpu().numpy()

            # save predictions & labels
            with open(preds_file_path, "w", encoding='utf-8') as f:
                json.dump(
                    {
                        "B_sus_preds": B_sus.tolist(),
                        "B_hold_out_preds": B_hold_out.tolist(),
                        "B_sus_labels": sus_label_B.cpu().numpy().tolist(),
                        "B_hold_out_labels": hold_out_label_B.cpu().numpy().tolist()
                    }, f, indent=4)

            # time end
            time_end = time.time()
            print(f"Time for experiment {test_i}: {time_end - time_start} seconds")

        print(f"Calculating significance for test {test_i}...")

        # use the robust log-space function
        p_value, neg_log_p_value, effect_size = t_test_with_effect(B_sus, B_hold_out)

        # save p-value
        with open(p_values_file_path, "w", encoding='utf-8') as f:
            json.dump({
                "p_value": p_value, 
                "log_p_value": neg_log_p_value,  
                "effect_size": effect_size
            }, f, indent=4)

        print(f"Result {test_i}: log_p={neg_log_p_value:.4f}, d={effect_size:.4f}")

    # aggragate p-values
    p_combined, combined_log_p, d_combined = aggregate_p_values(TTest_OUTPUT_DIR)

    with open(os.path.join(TTest_OUTPUT_DIR, "p_combined.json"), "w", encoding='utf-8') as f:
        json.dump({
            "p_combined": p_combined,
            "log_p_combined": combined_log_p,
            "effect_size_combined": d_combined,
            }, f, indent=4)
    print(f"p_combined: {p_combined}\t\tlog_p_combined (ln): {combined_log_p}\t\teffect_size_combined: {d_combined}")
