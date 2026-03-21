import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
from exp_utils.config_util.attack_config import attack_args, OUTPUT_ROOT_PATH, HOLD_OUT_OUTPUT_DIR
from exp_utils.dataset_process_util.emp_null_sample import OUTPUT_ROOT as METADATA_OUTPUT_ROOT

def load_hold_dataset_features(dataset_name):
    """
    Load pre-extracted MIA features for a single hold dataset
    """
    # TODO: You need to implement how the three hold datasets' features are stored
    # For example, if they are stored in the same format as extract_features.py
    FEATURE_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_PATH, "offline_features")
    
    # This is just an example - adjust according to your actual storage
    hold_feat_path = os.path.join(FEATURE_OUTPUT_DIR, f"{dataset_name}_features_full.pt")
    if os.path.exists(hold_feat_path):
        return torch.load(hold_feat_path)
    else:
        raise FileNotFoundError(f"Feature file not found: {hold_feat_path}")

def load_metadata_pair(pair_index):
    """
    Load metadata for a single base12 pair
    """
    base1_file = os.path.join(METADATA_OUTPUT_ROOT, f"pair_{pair_index}_base1.json")
    base2_file = os.path.join(METADATA_OUTPUT_ROOT, f"pair_{pair_index}_base2.json")
    
    with open(base1_file, 'r', encoding='utf-8') as f:
        base1_metadata = json.load(f)
    
    with open(base2_file, 'r', encoding='utf-8') as f:
        base2_metadata = json.load(f)
    
    return base1_metadata, base2_metadata

def collect_features_from_metadata(metadata_list, hold_dataset_features):
    """
    Collect features based on metadata
    hold_dataset_features should be a dict: {dataset_name: feature_tensor}
    """
    collected_features = []
    for item in metadata_list:
        source_dataset = item['source_dataset']
        original_index = item['original_index']
        
        if source_dataset not in hold_dataset_features:
            raise ValueError(f"Features for dataset {source_dataset} not found")
        
        feature = hold_dataset_features[source_dataset][original_index]
        collected_features.append(feature)
    
    return torch.stack(collected_features, dim=0)

def main():
    print(f"Metadata output directory: {METADATA_OUTPUT_ROOT}")
    
    # Example: Process one pair
    pair_idx = 0
    base1_meta, base2_meta = load_metadata_pair(pair_idx)
    
    print(f"Pair {pair_idx}:")
    print(f"  Base 1 has {len(base1_meta)} samples")
    print(f"  Base 2 has {len(base2_meta)} samples")
    print(f"  Sample Base 1 metadata: {base1_meta[0]}")
    
    # TODO: Load your hold dataset features here
    # hold_features = {
    #     'coco2017': load_hold_dataset_features('coco2017'),
    #     'nocaps': load_hold_dataset_features('nocaps'),
    #     'laion_multi': load_hold_dataset_features('laion_multi')
    # }
    
    # Then you can collect features:
    # base1_features = collect_features_from_metadata(base1_meta, hold_features)
    # base2_features = collect_features_from_metadata(base2_meta, hold_features)
    
    # Then use these features in your ttest
    print("\nYou can now load features for each sample from the original hold datasets")

if __name__ == "__main__":
    main()
