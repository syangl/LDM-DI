import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args, TRAIN_OUTPUT_DIR, dataset_config_1, HOLD_OUT_OUTPUT_DIR, dataset_config_2, PIPE_TYPE, OUTPUT_ROOT_PATH
from exp_utils.load_util.load_data import create_dataset_dataloader, get_img_dataLoader
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select

import torch
import numpy as np
import json

def exec_mia(method, pipe, sus_dataloader, hold_out_dataloader, choice=''):
    """
    Modified to extract features for the full dataset offline.
    """
    split = "full"
    
    if method == "structuralmi":
        print("Executing StructuralMI...")
        sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"structuralmi_{choice}_member_scores_{split}.json")
        hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"structuralmi_{choice}_nonmember_scores_{split}.json")
        if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/structuralmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.structuralmi_src.structuralmi import execute_structuralmi_attack
            
            sus_scores = execute_structuralmi_attack(dataloader=sus_dataloader, pipe=pipe, choice=choice)
            hold_out_scores = execute_structuralmi_attack(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(sus_scores_file), exist_ok=True)
            with open(sus_scores_file, "w", encoding='utf-8') as f: json.dump(sus_scores, f)
            os.makedirs(os.path.dirname(hold_out_scores_file), exist_ok=True)
            with open(hold_out_scores_file, "w", encoding='utf-8') as f: json.dump(hold_out_scores, f)
        else:
            print(f"{sus_scores_file} already exists, skipping inference.")
            with open(sus_scores_file, "r", encoding='utf-8') as f: sus_scores = json.load(f)
            with open(hold_out_scores_file, "r", encoding='utf-8') as f: hold_out_scores = json.load(f)
            
    elif method == "secmi":
        print("Executing SecMI...")
        sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"secmi_member_scores_{split}.json")
        hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"secmi_nonmember_scores_{split}.json")
        if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/secmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.secmi_src.secmi import get_reverse_denoise_results
            
            sus_scores = get_reverse_denoise_results(pipe=pipe, dataloader=sus_dataloader).cpu().numpy().tolist()
            sus_scores = [-score for score in sus_scores]
            hold_out_scores = get_reverse_denoise_results(pipe=pipe, dataloader=hold_out_dataloader).cpu().numpy().tolist()
            hold_out_scores = [-score for score in hold_out_scores]
            
            os.makedirs(os.path.dirname(sus_scores_file), exist_ok=True)
            with open(sus_scores_file, "w", encoding='utf-8') as f: json.dump(sus_scores, f)
            os.makedirs(os.path.dirname(hold_out_scores_file), exist_ok=True)
            with open(hold_out_scores_file, "w", encoding='utf-8') as f: json.dump(hold_out_scores, f)
        else:
            print(f"{sus_scores_file} already exists, skipping inference.")
            with open(sus_scores_file, "r", encoding='utf-8') as f: sus_scores = json.load(f)
            with open(hold_out_scores_file, "r", encoding='utf-8') as f: hold_out_scores = json.load(f)

    elif method == "naivelossmi":
        print("Executing NaiveLossMI...")
        sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"naivelossmi_member_scores_{split}.json")
        hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"naivelossmi_nonmember_scores_{split}.json")
        if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/naivelossmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.naivelossmi_src.naivelossmi import execute_naivelossmi_attack
            
            sus_scores = execute_naivelossmi_attack(dataloader=sus_dataloader, pipe=pipe)
            hold_out_scores = execute_naivelossmi_attack(dataloader=hold_out_dataloader, pipe=pipe)
            
            os.makedirs(os.path.dirname(sus_scores_file), exist_ok=True)
            with open(sus_scores_file, "w", encoding='utf-8') as f: json.dump(sus_scores, f)
            os.makedirs(os.path.dirname(hold_out_scores_file), exist_ok=True)
            with open(hold_out_scores_file, "w", encoding='utf-8') as f: json.dump(hold_out_scores, f)
        else:
            print(f"{sus_scores_file} already exists, skipping inference.")
            with open(sus_scores_file, "r", encoding='utf-8') as f: sus_scores = json.load(f)
            with open(hold_out_scores_file, "r", encoding='utf-8') as f: hold_out_scores = json.load(f)

    elif method == "piami":
        print("Executing PiAMI...")
        sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"piami_member_scores_{split}.json")
        hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"piami_nonmember_scores_{split}.json")
        if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/piami_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.piami_src.piami import execute_piami_attack
            
            sus_scores = execute_piami_attack(dataloader=sus_dataloader, pipe=pipe)
            hold_out_scores = execute_piami_attack(dataloader=hold_out_dataloader, pipe=pipe)
            
            os.makedirs(os.path.dirname(sus_scores_file), exist_ok=True)
            with open(sus_scores_file, "w", encoding='utf-8') as f: json.dump(sus_scores, f)
            os.makedirs(os.path.dirname(hold_out_scores_file), exist_ok=True)
            with open(hold_out_scores_file, "w", encoding='utf-8') as f: json.dump(hold_out_scores, f)
        else:
            print(f"{sus_scores_file} already exists, skipping inference.")
            with open(sus_scores_file, "r", encoding='utf-8') as f: sus_scores = json.load(f)
            with open(hold_out_scores_file, "r", encoding='utf-8') as f: hold_out_scores = json.load(f)

    elif method == "rediffusemi":
        print("Executing RediffuseMI...")
        sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"rediffusemi_{choice}_member_scores_{split}.json")
        hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"rediffusemi_{choice}_nonmember_scores_{split}.json")
        if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/rediffusemi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.rediffusemi_src.rediffusemi import execute_rediffusemi_attack
            
            sus_scores = execute_rediffusemi_attack(dataloader=sus_dataloader, pipe=pipe, choice=choice)
            hold_out_scores = execute_rediffusemi_attack(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(sus_scores_file), exist_ok=True)
            with open(sus_scores_file, "w", encoding='utf-8') as f: json.dump(sus_scores, f)
            os.makedirs(os.path.dirname(hold_out_scores_file), exist_ok=True)
            with open(hold_out_scores_file, "w", encoding='utf-8') as f: json.dump(hold_out_scores, f)
        else:
            print(f"{sus_scores_file} already exists, skipping inference.")
            with open(sus_scores_file, "r", encoding='utf-8') as f: sus_scores = json.load(f)
            with open(hold_out_scores_file, "r", encoding='utf-8') as f: hold_out_scores = json.load(f)

    elif method == "hintmi":
        print("Executing HintMI...")
        sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"hintmi_{choice}_member_scores_{split}.json")
        hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"hintmi_{choice}_nonmember_scores_{split}.json")
        if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/ours_hintmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.ours_hintmi_src.ours_hintmi import execute_hintmi
            
            sus_scores = execute_hintmi(dataloader=sus_dataloader, pipe=pipe, choice=choice)
            hold_out_scores = execute_hintmi(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(sus_scores_file), exist_ok=True)
            with open(sus_scores_file, "w", encoding='utf-8') as f: json.dump(sus_scores, f)
            os.makedirs(os.path.dirname(hold_out_scores_file), exist_ok=True)
            with open(hold_out_scores_file, "w", encoding='utf-8') as f: json.dump(hold_out_scores, f)
        else:
            print(f"{sus_scores_file} already exists, skipping inference.")
            with open(sus_scores_file, "r", encoding='utf-8') as f: sus_scores = json.load(f)
            with open(hold_out_scores_file, "r", encoding='utf-8') as f: hold_out_scores = json.load(f)

    elif method == "semansim":
        print("Executing Prompt Sensitivity...")
        sus_scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"semansim_member_scores_{split}.json")
        hold_out_scores_file = os.path.join(HOLD_OUT_OUTPUT_DIR, method, f"semansim_nonmember_scores_{split}.json")
        if not (os.path.exists(sus_scores_file) and os.path.exists(hold_out_scores_file)):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/ours_semansim_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.ours_semansim_src.ours_semansim import execute_semansim
            
            sus_scores = execute_semansim(dataloader=sus_dataloader, pipe=pipe, choice=choice)
            hold_out_scores = execute_semansim(dataloader=hold_out_dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(sus_scores_file), exist_ok=True)
            with open(sus_scores_file, "w", encoding='utf-8') as f: json.dump(sus_scores, f)
            os.makedirs(os.path.dirname(hold_out_scores_file), exist_ok=True)
            with open(hold_out_scores_file, "w", encoding='utf-8') as f: json.dump(hold_out_scores, f)
        else:
            print(f"{sus_scores_file} already exists, skipping inference.")
            with open(sus_scores_file, "r", encoding='utf-8') as f: sus_scores = json.load(f)
            with open(hold_out_scores_file, "r", encoding='utf-8') as f: hold_out_scores = json.load(f)
    else:
        raise ValueError(f"Unknown MIA method {method}")
        
    print(f"Member scores mean: {np.mean(sus_scores)}")
    print(f"Non-member scores mean: {np.mean(hold_out_scores)}\n")
    return sus_scores, hold_out_scores

def convert2tensor32(mia_scores):
    if hasattr(mia_scores, 'cpu'): 
        return mia_scores.cpu().float()
    else: 
        return torch.from_numpy(np.array(mia_scores, dtype=np.float32))

if __name__ == "__main__":
    FEATURE_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "offline_features")
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
    sus_feat_path = os.path.join(FEATURE_OUTPUT_DIR, "sus_features_full.pt")
    hold_feat_path = os.path.join(FEATURE_OUTPUT_DIR, "hold_out_features_full.pt")

    if os.path.exists(sus_feat_path) and os.path.exists(hold_feat_path):
        print("Full feature tensors already exist! You can run step2_main_ttest.py directly.")
        sys.exit(0)

    # load pipeline
    print("Loading pipeline for full dataset feature extraction...")
    pipe = txt2img_pipeline_select(base_model_id=attack_args.base_model_id, PIPE_TYPE=PIPE_TYPE)
    
    # load the full dataset
    _, suspect_ds = create_dataset_dataloader(
        output_dir=TRAIN_OUTPUT_DIR, 
        dataset_config=dataset_config_1,
        tokenizer=pipe.tokenizer, 
        img_resolution=attack_args.generate_resolution, 
        need_save=attack_args.is_visualize)
    _, hold_out_ds = create_dataset_dataloader(
        output_dir=HOLD_OUT_OUTPUT_DIR, dataset_config=dataset_config_2,
        tokenizer=pipe.tokenizer, img_resolution=attack_args.generate_resolution, need_save=attack_args.is_visualize)
    
    sus_dataloader = get_img_dataLoader(suspect_ds)
    hold_out_dataloader = get_img_dataLoader(hold_out_ds)

    # extract Features for all methods
    features_sus_list, features_hold_list = [], []
    
    methods = [
        ("secmi", ""), ("naivelossmi", ""), ("piami", ""), 
        ("structuralmi", "ssim"), ("rediffusemi", "ssim"), 
        ("hintmi", "ssim"), ("hintmi", "ssim_center"), 
        ("hintmi", "ssim_entropy"), ("semansim", "cos_sim")
    ]
    
    for method, choice in methods:
        sus_scores, hold_scores = exec_mia(method, pipe, sus_dataloader, hold_out_dataloader, choice)
        features_sus_list.append(convert2tensor32(sus_scores).unsqueeze(1))
        features_hold_list.append(convert2tensor32(hold_scores).unsqueeze(1))

    # concat and save raw features
    sus_features_full = torch.cat(features_sus_list, dim=1)
    hold_out_features_full = torch.cat(features_hold_list, dim=1)

    torch.save(sus_features_full, sus_feat_path)
    torch.save(hold_out_features_full, hold_feat_path)
    print(f"Successfully saved full features to {FEATURE_OUTPUT_DIR}")