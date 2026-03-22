import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
from exp_utils.config_util.attack_config import OUTPUT_ROOT_PATH
from exp_utils.dataset_process_util.emp_null_sample import OUTPUT_ROOT as METADATA_OUTPUT_ROOT


class MetadataFeatureLoader:
    """
    Unified loader class for metadata and MIA features
    """
    
    def __init__(self, feature_output_dir=None, metadata_output_dir=None):
        self.feature_output_dir = feature_output_dir or os.path.join(OUTPUT_ROOT_PATH, "offline_features")
        self.metadata_output_dir = metadata_output_dir or METADATA_OUTPUT_ROOT
        
        # initialize cached data
        self._feature_dict_cache = {}
        
        # original dataset names
        self.dataset_names = [
            "commoncatalog_cc_by",
            "laion_aesthetics_v2_5p0", 
            "coco_2017_val",
            "laion2b_multi_translated",
            "nocaps"
        ]
    
    def load_feature_dict(self, dataset_name):
        """
        Load feature dict for a dataset: {unique_id: feature_tensor}
        """
        if dataset_name in self._feature_dict_cache:
            return self._feature_dict_cache[dataset_name]
        
        feat_dict_path = os.path.join(self.feature_output_dir, f"{dataset_name}_features_dict.pt")
        if not os.path.exists(feat_dict_path):
            raise FileNotFoundError(f"Feature dict not found for dataset {dataset_name}: {feat_dict_path}")
        
        feature_dict = torch.load(feat_dict_path)
        self._feature_dict_cache[dataset_name] = feature_dict
        
        return feature_dict
    
    def load_all_datasets(self):
        """
        Preload all datasets' feature dicts
        Returns: {dataset_name: feature_dict}
        """
        result = {}
        for dataset_name in self.dataset_names:
            feature_dict = self.load_feature_dict(dataset_name)
            result[dataset_name] = feature_dict
        return result
    
    def get_feature_by_original_index(self, source_dataset, original_index):
        """
        Get a single sample's feature by (source_dataset, original_index)
        This is the unique identifier for each sample
        """
        feature_dict = self.load_feature_dict(source_dataset)
        unique_id = f"{source_dataset}_{original_index}"
        
        if unique_id not in feature_dict:
            raise KeyError(f"Unique ID {unique_id} not found")
        
        return feature_dict[unique_id]
    
    def collect_features_from_metadata_list(self, metadata_list):
        """
        Collect features from a list of metadata items
        Each item should have 'source_dataset' and 'original_index'
        Returns: torch.Tensor of shape [N, num_features]
        """
        features_list = []
        for item in metadata_list:
            source_dataset = item['source_dataset']
            original_index = item['original_index']
            feat = self.get_feature_by_original_index(source_dataset, original_index)
            features_list.append(feat.unsqueeze(0))
        
        return torch.cat(features_list, dim=0)
    
    def load_base12_pair(self, pair_index):
        """
        Load base1 and base2 metadata for a specific pair
        Returns: (base1_metadata_list, base2_metadata_list)
        """
        base1_file = os.path.join(self.metadata_output_dir, f"pair_{pair_index}_base1.json")
        base2_file = os.path.join(self.metadata_output_dir, f"pair_{pair_index}_base2.json")
        
        if not os.path.exists(base1_file) or not os.path.exists(base2_file):
            raise FileNotFoundError(f"Pair {pair_index} metadata not found")
        
        with open(base1_file, 'r', encoding='utf-8') as f:
            base1_metadata = json.load(f)
        
        with open(base2_file, 'r', encoding='utf-8') as f:
            base2_metadata = json.load(f)
        
        return base1_metadata, base2_metadata
    
    def load_ds12_features(self, pair_index):
        """
        Load ds1 and ds2 features for a specific pair
        Returns: (ds1_features_tensor, ds2_features_tensor)
        """
        ds1_meta, ds2_meta = self.load_ds12_pair(pair_index)
        
        ds1_features = self.collect_features_from_metadata_list(ds1_meta)
        ds2_features = self.collect_features_from_metadata_list(ds2_meta)
        
        return ds1_features, ds2_features


def create_loader():
    """
    Factory function to create a MetadataFeatureLoader instance
    """
    return MetadataFeatureLoader()


def main():
    """
    Example usage demonstration
    """
    loader = create_loader()
    
    print(f"\nMetadata output directory: {loader.metadata_output_dir}")
    print(f"Feature output directory: {loader.feature_output_dir}")
    
    # example 1: load a single pair metadata
    try:
        pair_idx = 0
        print(f"\n--- Loading pair {pair_idx} metadata ---")
        ds1_meta, ds2_meta = loader.load_ds12_pair(pair_idx)
        
        print(f"  DS 1 has {len(ds1_meta)} samples")
        print(f"  DS 2 has {len(ds2_meta)} samples")
        if ds1_meta:
            print(f"  Sample DS 1 metadata: {ds1_meta[0]}")
        if ds2_meta:
            print(f"  Sample DS 2 metadata: {ds2_meta[0]}")
        
        # example 2: load features for this pair
        print(f"\n--- Loading pair {pair_idx} features ---")
        ds1_feats, ds2_feats = loader.load_ds12_features(pair_idx)
        print(f"  DS 1 features shape: {ds1_feats.shape}")
        print(f"  DS 2 features shape: {ds2_feats.shape}")
        
        # example 3: get a single feature by (source_dataset, original_index)
        if ds1_meta:
            sample = ds1_meta[0]
            src_ds = sample['source_dataset']
            orig_idx = sample['original_index']
            print(f"\n--- Querying single feature ---")
            print(f"  Source dataset: {src_ds}")
            print(f"  Original index: {orig_idx}")
            
            feat = loader.get_feature_by_original_index(src_ds, orig_idx)
            print(f"  Feature shape: {feat.shape}")
            print(f"  Feature mean: {feat.mean().item():.4f}")
    
    except FileNotFoundError as e:
        print(f"\nNote: {e}")
        print("This is expected if you haven't generated the metadata and features yet.")
        print("Please run emp_null_sample.py and extract_features.py first.")


if __name__ == "__main__":
    main()
