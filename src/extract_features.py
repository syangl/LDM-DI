import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from exp_utils.config_util.attack_config import attack_args, PIPE_TYPE, OUTPUT_ROOT_PATH
if attack_args.experiment == "rq1":
    FEATURE_ROOT_PATH = os.path.join(OUTPUT_ROOT_PATH, "experiments", "RQ1_standard_di_vulnerability", "mia_features")
else:
    FEATURE_ROOT_PATH = os.path.join(OUTPUT_ROOT_PATH, "experiments", "shared_res", "mia_features")
from exp_utils.load_util.load_data import create_dataset_dataloader, get_img_dataLoader
from exp_utils.pipeline_util.txt2img_pipeline_select import txt2img_pipeline_select

import torch
import numpy as np
import json
import datetime
import time

def exec_mia(dataset_name, method, pipe, dataloader, choice=''):
    """
    Modified to extract features for the specified dataset offline.
    """
    split = "full"
    
    if method == "structuralmi":
        print("Executing StructuralMI...")
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"structuralmi_{choice}_scores_{split}.json")
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
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"secmi_scores_{split}.json")
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
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"naivelossmi_scores_{split}.json")
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
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"piami_scores_{split}.json")
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
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"rediffusemi_{choice}_scores_{split}.json")
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
    elif method == "gradient_maskingmi":
        print("Executing GradientMaskingMI...")
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"gradient_maskingmi_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/gradient_maskingmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.gradient_maskingmi_src.gradient_maskingmi import execute_gradient_maskingmi_attack
            
            scores = execute_gradient_maskingmi_attack(dataloader=dataloader, pipe=pipe)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)
    elif method == "multiple_lossmi":
        print("Executing MultipleLossMI...")
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"multiple_lossmi_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/multiple_lossmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.multiple_lossmi_src.multiple_lossmi import execute_multiple_lossmi_attack
            
            scores = execute_multiple_lossmi_attack(dataloader=dataloader, pipe=pipe)
            
            os.makedirs(os.path.dirname(scores_file), exist_ok=True)
            with open(scores_file, "w", encoding='utf-8') as f: 
                json.dump(scores, f)
        else:
            print(f"{scores_file} already exists, skipping inference.")
            with open(scores_file, "r", encoding='utf-8') as f: 
                scores = json.load(f)
    elif method == "noise_optimmi":
        print("Executing NoiseOptimMI...")
        scores_file = os.path.join(f"{FEATURE_ROOT_PATH}/offline_single_feature/{dataset_name}/{attack_args.model_name}/",
                                   f"noise_optimmi_scores_{split}.json")
        if not os.path.exists(scores_file):
            src_path = os.path.join(current_dir, "exp_utils/mi_util/noise_optimmi_src")
            if src_path not in sys.path: sys.path.insert(0, src_path)
            from exp_utils.mi_util.noise_optimmi_src.noise_optimmi import execute_noise_optimmi_attack
            
            scores = execute_noise_optimmi_attack(dataloader=dataloader, pipe=pipe)
            
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

def extract_features_for_dataset(dataset_config, pipe, feat_dict_path):
    """
    Extract features for a single dataset and save with unique identifiers
    """
    dataset_name = dataset_config["dataset"]
    print(f"\n{'='*60}")
    print(f"Extracting features for dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # load the dataset
    _, ds = create_dataset_dataloader(
        output_dir=None, 
        dataset_config=dataset_config,
        tokenizer=pipe.tokenizer, 
        img_resolution=attack_args.generate_resolution, 
        need_save=attack_args.is_visualize)
    
    dataloader = get_img_dataLoader(ds)
    
    # collect original indices
    original_indices = []
    for batch in dataloader:
        original_indices.extend(batch['index'])
    original_indices = [int(idx) for idx in original_indices]
    
    # extract Features for all methods
    features_list = []
    
    methods = [
        ("secmi", ""), 
        ("naivelossmi", ""), 
        ("piami", ""), 
        ("structuralmi", "ssim"), 
        ("rediffusemi", "ssim"), 
        ("gradient_maskingmi", ""), 
        ("multiple_lossmi", ""), 
        ("noise_optimmi", ""),
    ]
    
    for method, choice in methods:
        scores = exec_mia(dataset_name, method, pipe, dataloader, choice)
        if method == "noise_optimmi":
            # noise_optimmi returns two features per sample: [loss, l2]
            scores_np = np.array(scores)
            features_list.append(convert2tensor32(scores_np[:, 0]).unsqueeze(1))  # loss feature
            features_list.append(convert2tensor32(scores_np[:, 1]).unsqueeze(1))  # l2 feature
        elif method in ["gradient_maskingmi", "multiple_lossmi"]:
            # these methods return multiple timestep features per sample
            scores_np = np.array(scores)
            # add each timestep as a separate feature
            for i in range(scores_np.shape[1]):
                features_list.append(convert2tensor32(scores_np[:, i]).unsqueeze(1))
        else:
            # all other methods return one feature per sample
            features_list.append(convert2tensor32(scores).unsqueeze(1))
    
    # concat features
    features_full = torch.cat(features_list, dim=1)
    
    # create feature dict: {unique_identifier: feature_tensor}
    features_dict = {}
    for pos, original_idx in enumerate(original_indices):
        unique_id = f"{dataset_name}_{original_idx}"
        features_dict[unique_id] = features_full[pos]
    
    # save feature dict
    assert os.path.exists(os.path.dirname(feat_dict_path)), f"Feature dict path {feat_dict_path} does not exist"
    torch.save(features_dict, feat_dict_path)
    print(f"Successfully saved features dict for {dataset_name} to {feat_dict_path}")
    
    return features_full, features_dict

if __name__ == "__main__":
    torch.manual_seed(attack_args.global_seed)
    np.random.seed(attack_args.global_seed)
    FEATURE_OUTPUT_DIR = os.path.join(FEATURE_ROOT_PATH, "offline_features")
    os.makedirs(FEATURE_OUTPUT_DIR, exist_ok=True)
    
    # 10% sampling flag
    USE_10_PERCENT_SAMPLING = False
    SAMPLING_RATIO = 1.0
    
    # original datasets
    dataset_names = [
        "commoncatalog_cc_by_sa", # used for commoncanvas
        "coco_2017_val",
        "laion2b_multi_translated",
        "nocaps"
    ] if attack_args.model_name == "commoncanvas-s-c" else [
        "laion_aesthetics_v2_5p0", # used for stable-diffusion kandinsky
        "coco_2017_val",
        "laion2b_multi_translated",
        "nocaps"
    ]
    
    # load pipeline once
    print("Loading pipeline for feature extraction...")
    pipe = txt2img_pipeline_select(base_model_id=attack_args.base_model_id, PIPE_TYPE=PIPE_TYPE)
    
    # time tracking
    time_stats = {
        "model_name": attack_args.model_name,
        "start_time": datetime.datetime.now().isoformat(),
        "use_10_percent_sampling": USE_10_PERCENT_SAMPLING,
        "sampling_ratio": SAMPLING_RATIO,
        "datasets": {},
        "cached_datasets": [],
        "total_time_seconds": 0,
        "estimated_total_time_seconds": 0
    }
    
    # process each dataset
    for dataset_name in dataset_names:
        feat_dict_path = os.path.join(FEATURE_OUTPUT_DIR, f"{dataset_name}_{attack_args.model_name}_features_dict.pt")
        
        if os.path.exists(feat_dict_path):
            time_stats["cached_datasets"].append(dataset_name)
            print(f"Feature dict for {dataset_name} already exists, Skipping extraction.")
            continue
        
        # record start time
        dataset_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Starting processing dataset: {dataset_name}")
        print(f"Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if USE_10_PERCENT_SAMPLING:
            print(f"Mode: {int(SAMPLING_RATIO*100)}% sampling (estimated time x{1/SAMPLING_RATIO:.0f})")
        print(f"{'='*60}")
        
        # create dataset config
        dataset_split = "train" if dataset_name in ["commoncatalog_cc_by_sa", "laion_aesthetics_v2_5p0"] else "val"
        original_load_num = 1000 if dataset_name in ["commoncatalog_cc_by_sa", "laion_aesthetics_v2_5p0"] else 3000
        load_num = int(original_load_num * SAMPLING_RATIO) if USE_10_PERCENT_SAMPLING else original_load_num
        
        dataset_config = {
            "dataset": dataset_name,
            "load_path": f"{attack_args.local_dataset_path}/{dataset_name}/data/{dataset_split}",
            "load_num": load_num,
            "split": dataset_split,
        }
        
        # extract features for this dataset
        extract_features_for_dataset(dataset_config, pipe, feat_dict_path)
        
        # record end time
        dataset_end_time = time.time()
        dataset_elapsed = dataset_end_time - dataset_start_time
        
        # estimated time for full dataset
        estimated_elapsed = dataset_elapsed / SAMPLING_RATIO if USE_10_PERCENT_SAMPLING else dataset_elapsed
        
        # format time
        hours, rem = divmod(dataset_elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        
        est_hours, est_rem = divmod(estimated_elapsed, 3600)
        est_minutes, est_seconds = divmod(est_rem, 60)
        
        print(f"\n{'='*60}")
        print(f"Finished processing dataset: {dataset_name}")
        print(f"End time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Actual elapsed time ({load_num} samples): {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        if USE_10_PERCENT_SAMPLING:
            print(f"Estimated full dataset time ({original_load_num} samples): {int(est_hours)}h {int(est_minutes)}m {est_seconds:.2f}s")
        print(f"{'='*60}")
        
        # save stats
        time_stats["datasets"][dataset_name] = {
            "start_time": datetime.datetime.fromtimestamp(dataset_start_time).isoformat(),
            "end_time": datetime.datetime.fromtimestamp(dataset_end_time).isoformat(),
            "actual_elapsed_seconds": dataset_elapsed,
            "actual_elapsed_formatted": f"{int(hours)}h {int(minutes)}m {seconds:.2f}s",
            "estimated_elapsed_seconds": estimated_elapsed,
            "estimated_elapsed_formatted": f"{int(est_hours)}h {int(est_minutes)}m {est_seconds:.2f}s",
            "actual_sample_count": load_num,
            "full_sample_count": original_load_num
        }
        time_stats["total_time_seconds"] += dataset_elapsed
        time_stats["estimated_total_time_seconds"] += estimated_elapsed
    
    # calculate total time
    time_stats["end_time"] = datetime.datetime.now().isoformat()
    
    total_hours, total_rem = divmod(time_stats["total_time_seconds"], 3600)
    total_minutes, total_seconds = divmod(total_rem, 60)
    time_stats["total_time_formatted"] = f"{int(total_hours)}h {int(total_minutes)}m {total_seconds:.2f}s"
    
    est_total_hours, est_total_rem = divmod(time_stats["estimated_total_time_seconds"], 3600)
    est_total_minutes, est_total_seconds = divmod(est_total_rem, 60)
    time_stats["estimated_total_time_formatted"] = f"{int(est_total_hours)}h {int(est_total_minutes)}m {est_total_seconds:.2f}s"
    
    # save time stats to JSON
    time_stats_file = os.path.join(FEATURE_OUTPUT_DIR, f"time_stats_{attack_args.model_name}.json")
    with open(time_stats_file, "w", encoding='utf-8') as f:
        json.dump(time_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Actual total time: {time_stats['total_time_formatted']}")
    if USE_10_PERCENT_SAMPLING:
        print(f"Estimated full dataset total time: {time_stats['estimated_total_time_formatted']}")
    print(f"Time stats saved to: {time_stats_file}")
    print(f"{'='*60}")
    
