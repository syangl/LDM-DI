import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.config_util.attack_config import attack_args, TRAIN_OUTPUT_DIR, dataset_config_1, OUTPUT_ROOT_PATH
import json
import clip
from transformers import CLIPTokenizer
D_HOLD_OUT_NAME = f"D_hold_out_{dataset_config_1['dataset']}"
D_HOLD_OUT_OUTPUT_ROOT = os.path.join(attack_args.local_dataset_path, D_HOLD_OUT_NAME, "metadata")

class DistributionOverlapFinder:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # preprocess
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.features1 = None

    def extract_features(self, image_tensors, batch_size=64):
        features = []

        with torch.no_grad():
            for i in tqdm(range(0, len(image_tensors), batch_size), desc="Extracting features"):
                batch = image_tensors[i:i + batch_size]
                batch = torch.stack(batch).to(self.device)
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                features.append(feats.cpu().numpy())

        return np.vstack(features)

    def compute_overlap_scores(self, features2):
        """
        Compute overlap scores between two feature matrices.
        """
        k = 5
        sims = self.features1 @ features2.T
        sims21 = np.sort(sims.T, axis=1)[:, -k:]
        overlap_scores = sims21.mean(axis=1)
        return overlap_scores

    def process_single_dataset2(self, dataloader2, dataset_info):
        """
        Process single dataset2 and compute overlap scores.
        Returns: (scores, sample_infos)
        """
        dataset2 = []
        for batch in tqdm(dataloader2, desc=f"Loading {dataset_info['name']}"):
            images = batch['image']
            indexes = batch['index']
            for i in range(len(images)):
                img_tensor = images[i]
                dataset2.append((indexes[i], img_tensor))
        
        img_tensors2 = [item[1] for item in dataset2]
        features2 = self.extract_features(img_tensors2)
        scores = self.compute_overlap_scores(features2)
        
        sample_infos = []
        for i, (index, img_tensor) in enumerate(dataset2):
            sample_infos.append({
                'local_index': index,
                'dataset_name': dataset_info['name'],
                'load_path': dataset_info['load_path'],
                'split': dataset_info['split'],
                'score': scores[i]
            })
        
        del img_tensors2, features2, dataset2
        import gc
        gc.collect()
        
        return scores, sample_infos

    def save_metadata_only(self, overlap_dataset_with_info):
        os.makedirs(D_HOLD_OUT_OUTPUT_ROOT, exist_ok=True)
        
        metadata_list = []
        for new_index, (local_index, _, info) in enumerate(overlap_dataset_with_info):
            metadata_list.append({
                'source_dataset': info['dataset_name'],
                'original_index': int(local_index)
            })
        
        output_file = os.path.join(D_HOLD_OUT_OUTPUT_ROOT, f"meta_{D_HOLD_OUT_NAME}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4)
        
        print(f"Saved {len(metadata_list)} samples metadata to {output_file}")


def find_distribution_overlap(dataloader1, dataloaders2, overlap_finder, target_num):
    """
    Find examples in mixed dataloaders2 with maximum distribution overlap with dataloader1.
    """
    # process dataset 1
    print("Processing dataset 1...")
    dataset1 = []
    for batch in tqdm(dataloader1, desc="Loading dataset 1"):
        images = batch['image']
        indexes = batch['index']
        for i in range(len(images)):
            img_tensor = images[i]
            dataset1.append((indexes[i], img_tensor))
    
    img_tensors1 = [item[1] for item in dataset1]
    overlap_finder.features1 = overlap_finder.extract_features(img_tensors1)
    
    del img_tensors1, dataset1
    import gc
    gc.collect()
    
    print(f"Features for dataset 1 extracted, shape: {overlap_finder.features1.shape}")
    
    # process all dataloaders2
    all_sample_infos = []
    
    for dataset_idx, (dataloader2, dataset2_name, dataset2_load_path, dataset2_split) in enumerate(dataloaders2):
        print(f"\nProcessing dataset 2 [{dataset_idx+1}/{len(dataloaders2)}] - {dataset2_name}")
        
        dataset_info = {
            'name': dataset2_name,
            'load_path': dataset2_load_path,
            'split': dataset2_split
        }
        
        scores, sample_infos = overlap_finder.process_single_dataset2(dataloader2, dataset_info)
        all_sample_infos.extend(sample_infos)
        
        print(f"  - Processed {len(sample_infos)} samples from {dataset2_name}")
    
    # select top target_num samples
    print(f"\nSelecting top {target_num} samples from {len(all_sample_infos)} total samples...")
    all_sample_infos.sort(key=lambda x: x['score'], reverse=True)
    top_samples = all_sample_infos[:target_num]
    
    # build return format for save_overlap_images
    overlap_dataset2_with_info = []
    for sample_info in top_samples:
        overlap_dataset2_with_info.append((
            sample_info['local_index'],
            None,  
            sample_info
        ))
    
    print(f"Found {len(overlap_dataset2_with_info)} overlap examples in mixed search dataset")
    
    overlap_finder.features1 = None
    gc.collect()
    
    return overlap_dataset2_with_info


def find_overlap_main(
    dataloader1,
    dataloaders2,
    target_num,
):
    # find overlapping examples
    overlap_finder = DistributionOverlapFinder()
    overlap_holdout = find_distribution_overlap(dataloader1, dataloaders2, overlap_finder, target_num)

    overlap_finder.save_metadata_only(overlap_dataset_with_info=overlap_holdout)

    print("Overlap metadata saved")


if __name__ == "__main__":
    TARGET_NUM = attack_args.dataset_total_num
    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(attack_args.base_model_id, "tokenizer"))

    # dataset 1
    dataloader1, _ = create_dataset_dataloader(TRAIN_OUTPUT_DIR,
                                               dataset_config_1,
                                               tokenizer,
                                               224, 
                                               64,
                                               need_save=False)
    
    # given hold-out source data Q1, Q2, ...
    dataset_configs_2 = [
        # Q1
        {
            "dataset": "coco_2017_val", 
            "load_path": f"{attack_args.local_dataset_path}/coco_2017_val/data/val",
            "load_num": 3000,  # equal sub source proportion to each Q_i
            "split": "val",
        },
        # Q2
        {
            "dataset": "nocaps",  
            "load_path": f"{attack_args.local_dataset_path}/nocaps/data/val",
            "load_num": 3000,
            "split": "val",
        },
        # Q3
        {
            "dataset": "laion2b_multi_translated",  
            "load_path": f"{attack_args.local_dataset_path}/laion2b_multi_translated/data/val",
            "load_num": 3000,
            "split": "val",
        },
    ]
    
    # load all source datasets
    dataloaders2 = []
    for config in dataset_configs_2:
        temp_output_dir = f"{OUTPUT_ROOT_PATH}/output/{config['dataset']}/{attack_args.model_name}/"
        dataloader, _ = create_dataset_dataloader(temp_output_dir,
                                                   config,
                                                   tokenizer,
                                                   224,
                                                   64,
                                                   need_save=False)
        dataloaders2.append((dataloader, config['dataset'], config['load_path'], config['split']))

    # get TARGET_NUM images from Q = {Q1, Q2, Q3}
    find_overlap_main(dataloader1=dataloader1,
                      dataloaders2=dataloaders2,
                      target_num=TARGET_NUM,
                      )
