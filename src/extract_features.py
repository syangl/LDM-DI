import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args, TRAIN_OUTPUT_DIR, dataset_config_1, PIPE_TYPE, OUTPUT_ROOT_PATH
from exp_utils.load_util.load_data import create_dataset_dataloader, get_img_dataLoader
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select

import torch
import numpy as np
import json

def exec_mia(method, pipe, dataloader, choice=''):
    """
    Modified to extract features for the specified dataset offline.
    """
    split = "full"
    
    if method == "structuralmi":
        print("Executing StructuralMI...")
        scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"structuralmi_{choice}_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/structuralmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.structuralmi_src.structuralmi import execute_structuralmi_attack
            
            scores = execute_structuralmi_attack(dataloader=dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)
            
    elif method == "secmi":
        print("Executing SecMI...")
        scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"secmi_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/secmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.secmi_src.secmi import get_reverse_denoise_results
            
            scores = get_reverse_denoise_results(pipe=pipe, dataloader=dataloader).cpu().numpy().tolist()
            scores = [-score for score in scores]
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)

    elif method == "naivelossmi":
        print("Executing NaiveLossMI...")
        scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"naivelossmi_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/naivelossmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.naivelossmi_src.naivelossmi import execute_naivelossmi_attack
            
            scores = execute_naivelossmi_attack(dataloader=dataloader, pipe=pipe)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)

    elif method == "piami":
        print("Executing PiAMI...")
        scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"piami_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/piami_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.piami_src.piami import execute_piami_attack
            
            scores = execute_piami_attack(dataloader=dataloader, pipe=pipe)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)

    elif method == "rediffusemi":
        print("Executing RediffuseMI...")
        scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"rediffusemi_{choice}_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/rediffusemi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.rediffusemi_src.rediffusemi import execute_rediffusemi_attack
            
            scores = execute_rediffusemi_attack(dataloader=dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)

    elif method == "hintmi":
        print("Executing HintMI...")
        scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"hintmi_{choice}_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/ours_hintmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.ours_hintmi_src.ours_hintmi import execute_hintmi
            
            scores = execute_hintmi(dataloader=dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)

    elif method == "semansim":
        print("Executing Prompt Sensitivity...")
        scores_file = os.path.join(TRAIN_OUTPUT_DIR, method, f"semansim_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/ours_semansim_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.ours_semansim_src.ours_semansim import execute_semansim
            
            scores = execute_semansim(dataloader=dataloader, pipe=pipe, choice=choice)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)
    else:
        raise ValueError(f"Unknown MIA method {method}")
        
    print(f"Scores mean: {np.mean(scores)}\n")
    return scores

def convert2tensor32(mia_scores):
    if hasattr(mia_scores, 'cpu'): 
        return mia_scores.cpu().float()
    else: 
        return torch.from_numpy(np.array(mia_scores, dtype=np.float32))

if __name__ == "__main__":
    FEATURE_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "offline_features")
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
    feat_path = os.path.join(FEATURE_OUTPUT_DIR, f"{dataset_config_1['dataset']}_features_full.pt")

    if os.path.exists(feat_path):
        print(f"Feature tensor for {dataset_config_1['dataset']} already exists!")
        sys.exit(0)

    # load pipeline
    print(f"Loading pipeline for {dataset_config_1['dataset']} feature extraction...")
    pipe = txt2img_pipeline_select(base_model_id=attack_args.base_model_id, PIPE_TYPE=PIPE_TYPE)
    
    # load the dataset
    _, ds = create_dataset_dataloader(
        output_dir=TRAIN_OUTPUT_DIR, 
        dataset_config=dataset_config_1,
        tokenizer=pipe.tokenizer, 
        img_resolution=attack_args.generate_resolution, 
        need_save=attack_args.is_visualize)
    
    dataloader = get_img_dataLoader(ds)

    # extract Features for all methods
    features_list = []
    
    methods = [
        ("secmi", ""), ("naivelossmi", ""), ("piami", ""), 
        ("structuralmi", "ssim"), ("rediffusemi", "ssim"), 
        ("hintmi", "ssim"), ("hintmi", "ssim_center"), 
        ("hintmi", "ssim_entropy"), ("semansim", "cos_sim")
    ]
    
    for method, choice in methods:
        scores = exec_mia(method, pipe, dataloader, choice)
        features_list.append(convert2tensor32(scores).unsqueeze(1))

    # concat and save raw features
    features_full = torch.cat(features_list, dim=1)

    torch.save(features_full, feat_path)
    print(f"Successfully saved features for {dataset_config_1['dataset']} to {feat_path}")