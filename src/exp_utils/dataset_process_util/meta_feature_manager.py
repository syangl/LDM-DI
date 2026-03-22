import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import torch
from exp_utils.config_util.attack_config import OUTPUT_ROOT_PATH, attack_args
from exp_utils.dataset_process_util.emp_null_sample import OUTPUT_ROOT as D_BASE_METADATA_OUTPUT_ROOT
from exp_utils.dataset_process_util.clip_knn import D_HOLD_OUT_OUTPUT_ROOT as D_HOLD_OUT_METADATA_OUTPUT_ROOT

class MetadataFeatureLoader:
    """
    Unified loader class for metadata and MIA features
    """
    
    def __init__(self, feature_output_dir=None, d_base_metadata_output_dir=None, d_hold_out_metadata_output_dir=None):
        self.feature_output_dir = feature_output_dir or os.path.join(OUTPUT_ROOT_PATH, "offline_features")
        self.d_base_metadata_output_dir = d_base_metadata_output_dir or D_BASE_METADATA_OUTPUT_ROOT
        self.d_hold_out_metadata_output_dir = d_hold_out_metadata_output_dir or D_HOLD_OUT_METADATA_OUTPUT_ROOT
        
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
        
        feat_dict_path = os.path.join(self.feature_output_dir, f"{dataset_name}_{attack_args.model_name}_features_dict.pt")
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
        base1_file = os.path.join(self.d_base_metadata_output_dir, f"pair_{pair_index}_base1.json")
        base2_file = os.path.join(self.d_base_metadata_output_dir, f"pair_{pair_index}_base2.json")
        
        if not os.path.exists(base1_file) or not os.path.exists(base2_file):
            raise FileNotFoundError(f"Pair {pair_index} metadata not found")
        
        with open(base1_file, 'r', encoding='utf-8') as f:
            base1_metadata = json.load(f)
        
        with open(base2_file, 'r', encoding='utf-8') as f:
            base2_metadata = json.load(f)
        
        return base1_metadata, base2_metadata
    
    def load_base12_features(self, pair_index):
        """
        Load base1 and base2 features for a specific pair
        Returns: (base1_features_tensor, base2_features_tensor)
        """
        base1_meta, base2_meta = self.load_base12_pair(pair_index)
        
        base1_features = self.collect_features_from_metadata_list(base1_meta)
        base2_features = self.collect_features_from_metadata_list(base2_meta)
        
        return base1_features, base2_features
    
    def load_ds12_features(self, pair_index):
        """
        Alias for load_base12_features (backward compatibility)
        """
        return self.load_base12_features(pair_index)
    
    def load_ds12_pair(self, pair_index):
        """
        Alias for load_base12_pair (backward compatibility)
        """
        return self.load_base12_pair(pair_index)
    
    def load_hold_out_metadata(self, dataset_name):
        """
        Load hold-out dataset metadata
        Returns: metadata_list
        """
        hold_out_file = os.path.join(self.d_hold_out_metadata_output_dir, f"meta_{dataset_name}.json")
        
        if not os.path.exists(hold_out_file):
            raise FileNotFoundError(f"Hold-out dataset metadata not found for {dataset_name}: {hold_out_file}")
        
        with open(hold_out_file, 'r', encoding='utf-8') as f:
            hold_out_metadata = json.load(f)
        
        return hold_out_metadata
    
    def load_single_dataset_metadata(self, dataset_name):
        """
        Load metadata for a single dataset by name
        Determines whether it's base12 pair or hold-out dataset automatically
        """
        # Check if it's a hold-out dataset (starts with D_hold_out_)
        if dataset_name.startswith("D_hold_out_"):
            return self.load_hold_out_metadata(dataset_name)
        # Check if it's D_base_1 or D_base_2 (but these are part of pairs)
        elif dataset_name in ["D_base_1", "D_base_2"]:
            raise ValueError("D_base_1 and D_base_2 are only accessible as pairs, use load_base12_pair instead")
        else:
            # For other dataset names, try to load as single dataset metadata
            # First check overlap results
            hold_out_file = os.path.join(self.d_hold_out_metadata_output_dir, f"meta_{dataset_name}.json")
            if os.path.exists(hold_out_file):
                with open(hold_out_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # If not found, raise error
            raise FileNotFoundError(f"Metadata not found for dataset {dataset_name}")
    
    def load_dataset_pair_features(self, dataset1_name, dataset2_name, pair_index=None):
        """
        Load features for a pair of datasets
        Handles both base12 pairs and hold-out + base dataset pairs
        
        Args:
            dataset1_name: Name of first dataset
            dataset2_name: Name of second dataset
            pair_index: Optional pair index for base12 pairs
        """
        # Case 1: Both are D_base_1 and D_base_2 (base12 pair)
        if {dataset1_name, dataset2_name} == {"D_base_1", "D_base_2"}:
            if pair_index is None:
                raise ValueError("pair_index required for base12 pairs")
            ds1_feats, ds2_feats = self.load_base12_features(pair_index)
            if dataset1_name == "D_base_2":
                return ds2_feats, ds1_feats
            return ds1_feats, ds2_feats
        
        # Case 2: One is D_hold_out_*, the other is a regular dataset
        # Load each dataset separately
        ds1_meta = self.load_single_dataset_metadata(dataset1_name)
        ds2_meta = self.load_single_dataset_metadata(dataset2_name)
        
        ds1_features = self.collect_features_from_metadata_list(ds1_meta)
        ds2_features = self.collect_features_from_metadata_list(ds2_meta)
        
        return ds1_features, ds2_features


def create_loader():
    """
    Factory function to create a MetadataFeatureLoader instance
    """
    return MetadataFeatureLoader()


# def main():
#     """
#     Example usage demonstration
#     """
#     loader = create_loader()
    
#     print(f"\nMetadata output directory: {loader.metadata_output_dir}")
#     print(f"Feature output directory: {loader.feature_output_dir}")
    
#     # example 1: load a single pair metadata
#     try:
#         pair_idx = 0
#         print(f"\n--- Loading pair {pair_idx} metadata ---")
#         ds1_meta, ds2_meta = loader.load_ds12_pair(pair_idx)
        
#         print(f"  DS 1 has {len(ds1_meta)} samples")
#         print(f"  DS 2 has {len(ds2_meta)} samples")
#         if ds1_meta:
#             print(f"  Sample DS 1 metadata: {ds1_meta[0]}")
#         if ds2_meta:
#             print(f"  Sample DS 2 metadata: {ds2_meta[0]}")
        
#         # example 2: load features for this pair
#         print(f"\n--- Loading pair {pair_idx} features ---")
#         ds1_feats, ds2_feats = loader.load_ds12_features(pair_idx)
#         print(f"  DS 1 features shape: {ds1_feats.shape}")
#         print(f"  DS 2 features shape: {ds2_feats.shape}")
        
#         # example 3: get a single feature by (source_dataset, original_index)
#         if ds1_meta:
#             sample = ds1_meta[0]
#             src_ds = sample['source_dataset']
#             orig_idx = sample['original_index']
#             print(f"\n--- Querying single feature ---")
#             print(f"  Source dataset: {src_ds}")
#             print(f"  Original index: {orig_idx}")
            
#             feat = loader.get_feature_by_original_index(src_ds, orig_idx)
#             print(f"  Feature shape: {feat.shape}")
#             print(f"  Feature mean: {feat.mean().item():.4f}")
    
#     except FileNotFoundError as e:
#         print(f"\nNote: {e}")
#         print("This is expected if you haven't generated the metadata and features yet.")
#         print("Please run emp_null_sample.py and extract_features.py first.")


# if __name__ == "__main__":
#     main()
