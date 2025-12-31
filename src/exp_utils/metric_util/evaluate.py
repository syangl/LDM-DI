'''
Evaluation Metrics
'''
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc
import os
import json
from exp_utils.config_util.attack_config import attack_args
import matplotlib.pyplot as plt
from tqdm import tqdm
import clip
from skimage.measure import shannon_entropy
from exp_utils.dataset_process_util.img_preprocess import tensor2pil
import torch


def calculate_ssim(img_outs_list, img_ref_list):
    score_list = []
    mean_ssim = 0.0
    with tqdm(total=len(img_ref_list)) as tqdm_bar:
        # for each index, calculate SSIM of ref and outs
        for out_list, img_ref in zip(img_outs_list, img_ref_list):
            tmp_score = []
            for img_out in out_list:
                # convert to numpy array (RGB)
                img_out_np = np.array(img_out)
                img_ref_np = np.array(img_ref)

                # SSIM
                ssim_value = ssim(img_out_np,
                                img_ref_np,
                                data_range=2.0,
                                channel_axis=0).item()
                tmp_score.append(ssim_value)
            img_score = np.mean(tmp_score)
            score_list.append(img_score)

            # update progess bar
            mean_ssim += img_score
            tqdm_bar.set_description(f"mean ssim: {mean_ssim / len(score_list):.8f}")
            tqdm_bar.update(1)

    return score_list


def calculate_ssim_entropy(img_outs_list,
                           img_ref_list,
                           crop_resolution,
                           alpha=0.9,
                           beta=0.1,
                           ):
    _, h, w = img_ref_list[0].shape
    h_start = (h - crop_resolution) // 2
    w_start = (w - crop_resolution) // 2

    score_list = []
    mean_score = 0.0
    # for each index, calculate SSIM of ref and outs
    with tqdm(total=len(img_ref_list)) as tqdm_bar:
        for out_list, img_ref in zip(img_outs_list, img_ref_list):
            tmp_score = []

            img_ref_crop = img_ref[:, h_start:h_start + crop_resolution, w_start:w_start + crop_resolution]
            img_ref_np = np.array(img_ref_crop)

            ref_gray = img_ref_np.mean(axis=0)
            ref_entropy = shannon_entropy(ref_gray)

            for img_out in out_list:
                img_out_crop = img_out[:, h_start:h_start + crop_resolution, w_start:w_start + crop_resolution]
                # convert to numpy array (RGB)
                img_out_np = np.array(img_out_crop)

                out_gray = img_out_np.mean(axis=0)

                # SSIM
                ssim_value = ssim(img_out_np, img_ref_np, data_range=2.0, channel_axis=0).item()
                # entropy
                out_entropy = shannon_entropy(out_gray)
                entropy_score = 1 - abs(ref_entropy - out_entropy) / max(ref_entropy, 1e-6)
                total_score = alpha * ssim_value + beta * entropy_score
                tmp_score.append(float(total_score))

            img_score = np.mean(tmp_score)
            score_list.append(img_score)

            # update progess bar
            mean_score += img_score
            tqdm_bar.set_description(f"mean ssim-entropy: {mean_score / len(score_list):.8f}")
            tqdm_bar.update(1)

    return score_list


def calculate_ssim_center(img_outs_list, img_ref_list, crop_resolution):
    score_list = []
    mean_ssim = 0.0
    # crop parameters
    _, h, w = img_ref_list[0].shape
    h_start = (h - crop_resolution) // 2
    w_start = (w - crop_resolution) // 2
    # for each index, calculate SSIM of ref and outs
    with tqdm(total=len(img_ref_list)) as tqdm_bar:
        for out_list, img_ref in zip(img_outs_list, img_ref_list):
            # crop to avoid invalid border effect
            img_ref_crop = img_ref[:, h_start:h_start + crop_resolution, w_start:w_start + crop_resolution]
            img_ref_np = np.array(img_ref_crop)

            tmp_score = []
            for img_out in out_list:
                # crop to avoid invalid border effect
                img_out_crop = img_out[:, h_start:h_start + crop_resolution, w_start:w_start + crop_resolution]

                # convert to numpy array (RGB)
                img_out_np = np.array(img_out_crop)

                # SSIM
                ssim_value = ssim(img_out_np,
                                img_ref_np,
                                data_range=2.0,
                                channel_axis=0).item()
                # tmp_score.append(ssim_value * texture_fac)
                tmp_score.append(ssim_value)

            img_score = np.mean(tmp_score)
            score_list.append(img_score)

            # update progess bar
            mean_ssim += img_score
            tqdm_bar.set_description(f"mean center ssim: {mean_ssim / len(score_list):.8f}")
            tqdm_bar.update(1)

    return score_list



def calculate_cos_sim(img_outs_list, img_ref_list, clip_model, clip_preprocess):
    score_list = []
    mean_cos_sim = 0.0
    with tqdm(total=len(img_ref_list)) as tqdm_bar:
        # for each index, calculate cosine similarity of ref and outs
        for out_list, img_ref in zip(img_outs_list, img_ref_list):
            # encode reference image
            ref_input = clip_preprocess(tensor2pil(img_ref)).unsqueeze(0).to("cuda")
            with torch.no_grad():
                ref_feat = clip_model.encode_image(ref_input)
                ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)

            tmp_score = []
            for img_out in out_list:
                # encode output image
                out_input = clip_preprocess(tensor2pil(img_out)).unsqueeze(0).to("cuda")
                with torch.no_grad():
                    out_feat = clip_model.encode_image(out_input)
                    out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)

                cos_sim_value = (ref_feat @ out_feat.T).item()
                tmp_score.append(cos_sim_value)

            img_score = np.mean(tmp_score)
            score_list.append(img_score)

            # update progess bar
            mean_cos_sim += img_score
            tqdm_bar.set_description(f"mean cos sim: {mean_cos_sim / len(score_list):.8f}")
            tqdm_bar.update(1)

    return score_list



def calculate_similarity(img_outs_list, img_ref_list, choice, **opt_args):
    """
    Evaluate and save metrics
    """
    if choice == "ssim":
        score_list = calculate_ssim(img_outs_list, img_ref_list)
    elif choice == "ssim_center":
        score_list = calculate_ssim_center(img_outs_list, img_ref_list, int(attack_args.generate_resolution * 0.8))
    elif choice == "ssim_entropy":
        score_list = calculate_ssim_entropy(img_outs_list, img_ref_list, int(attack_args.generate_resolution * 0.8))
    elif choice == "cos_sim":
        prompt_embed_list = opt_args.get("prompt_embed_list", None)
        assert prompt_embed_list is not None, "prompt_embed_list is required for calculate cosine similarity"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
        score_list = calculate_cos_sim(img_outs_list, img_ref_list, clip_model, clip_preprocess)
    else:
        raise ValueError("Invalid choice.")

    return score_list



def evaluate_roc(train_scores, hold_out_scores, save_path, larger_is_member):
    """
    ROC curve evaluation (refer to code implementation of Duan et al.)
    """
    train_scores = np.array(train_scores)
    hold_out_scores = np.array(hold_out_scores)

    # get min max score
    min_score = min(train_scores.min(), hold_out_scores.min())
    max_score = max(train_scores.max(), hold_out_scores.max())

    TPR_list = []
    FPR_list = []
    acc_list = []
    threshold_list = np.arange(min_score, max_score, (max_score - min_score) / 10000)

    # total number of examples
    total = len(train_scores) + len(hold_out_scores)

    if larger_is_member: # if member usually has higher score
        for threshold in threshold_list:
            # TP, TN, FP, FN
            TP = (train_scores >= threshold).sum()
            TN = (hold_out_scores < threshold).sum()
            FP = (hold_out_scores >= threshold).sum()
            FN = (train_scores < threshold).sum()

            # accuracy
            acc = (TP + TN) / (TP + TN + FP + FN)
            acc_list.append(acc.item())

            # TPR, FPR
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            TPR_list.append(TPR.item())
            FPR_list.append(FPR.item())

    else: # if member usually has smaller scores
        for threshold in threshold_list:
            # TP, TN, FP, FN
            TP = (train_scores <= threshold).sum()
            TN = (hold_out_scores > threshold).sum()
            FP = (hold_out_scores <= threshold).sum()
            FN = (train_scores > threshold).sum()

            # accuracy = (TP + TN) / (TP + TN + FP + FN)
            acc = (TP + TN) / (TP + TN + FP + FN)
            acc_list.append(acc.item())

            # TPR, FPR
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            TPR_list.append(TPR.item())
            FPR_list.append(FPR.item())

    # TPR @ 1% FPR
    TPR_at_1_percent_FPR = TPR_list[np.argmin(np.abs(np.asarray(FPR_list) - 0.01))]
    # TPR @ 0.1 % FPR
    TPR_at_0_1_percent_FPR = TPR_list[np.argmin(np.abs(np.asarray(FPR_list) - 0.001))]

    # auc value
    auc_value = auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUC: {auc_value}')

    # save roc
    plt.figure(figsize=(8, 6))
    plt.plot(FPR_list, TPR_list, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], "k--")  # baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Membership Inference ROC Curve")
    plt.legend(loc="lower right", fontsize=16)
    plt.grid(True)
    plt.tick_params(axis="both", which="major", direction="in")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{attack_args.used_datasets[0]}_{attack_args.used_datasets[1]}--roc.png"), dpi=300)

    # save accuracy
    with open(os.path.join(save_path, f"{attack_args.used_datasets[0]}_{attack_args.used_datasets[1]}--acc.jsonl"), "w", encoding="utf-8") as f:
        for threshold, acc, TPR, FPR in zip(threshold_list, acc_list, TPR_list, FPR_list):
            json.dump(f"threshold: {threshold:.16f} \t ASR: {acc:.8f} \t TPR: {TPR:.8f} \t FPR: {FPR:.8f}", f)
            f.write("\n")

    # save final results
    with open(os.path.join(save_path, f"{attack_args.used_datasets[0]}_{attack_args.used_datasets[1]}--final.jsonl"), "w", encoding="utf-8") as f:
        result = {
            "Total number of examples": total,
            "ASR": np.max(acc_list),
            "AUC": auc_value,
            "TPR @ 1% FPR": TPR_at_1_percent_FPR,
            "TPR @ 0.1% FPR": TPR_at_0_1_percent_FPR,
        }
        json.dump(result, f)


def evaluate_all_roc(train_scores, hold_out_scores, larger_is_member):
    train_scores = np.array(train_scores)
    hold_out_scores = np.array(hold_out_scores)

    # get min max score
    min_score = min(train_scores.min(), hold_out_scores.min())
    max_score = max(train_scores.max(), hold_out_scores.max())

    TPR_list = []
    FPR_list = []
    acc_list = []
    threshold_list = np.arange(min_score, max_score, (max_score - min_score) / 10000)

    # total number of examples
    total = len(train_scores) + len(hold_out_scores)

    if larger_is_member: # if member usually has higher score
        for threshold in threshold_list:
            # TP, TN, FP, FN
            TP = (train_scores >= threshold).sum()
            TN = (hold_out_scores < threshold).sum()
            FP = (hold_out_scores >= threshold).sum()
            FN = (train_scores < threshold).sum()

            # accuracy
            acc = (TP + TN) / (TP + TN + FP + FN)
            acc_list.append(acc.item())

            # TPR, FPR
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)

            TPR_list.append(TPR.item())
            FPR_list.append(FPR.item())

    else: # if member usually has smaller scores
        for threshold in threshold_list:
            # TP, TN, FP, FN
            TP = (train_scores <= threshold).sum()
            TN = (hold_out_scores > threshold).sum()
            FP = (hold_out_scores <= threshold).sum()
            FN = (train_scores > threshold).sum()

            # accuracy = (TP + TN) / (TP + TN + FP + FN)
            acc = (TP + TN) / (TP + TN + FP + FN)
            acc_list.append(acc.item())

            # TPR, FPR
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            TPR_list.append(TPR.item())
            FPR_list.append(FPR.item())

    # auc value
    auc_value = auc(np.asarray(FPR_list), np.asarray(TPR_list))
    return FPR_list, TPR_list, auc_value
