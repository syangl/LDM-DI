import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.config_util.attack_config import attack_args, HOLD_OUT_OUTPUT_DIR, dataset_config_2

import json
import random
from tqdm import tqdm
from transformers import CLIPTokenizer


def save_split_metadata_only(
    dataset_split,
    original_dataset_name,
    split_name,
    output_dir="split_metadata"
):
    """
    Save only metadata for split datasets
    """
    print(f"Saving {split_name} metadata with {len(dataset_split)} samples...")

    # paths
    split_dataset_name = f"{original_dataset_name}_{split_name}"
    save_dir = os.path.join(current_dir, output_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Create metadata list with source_dataset and original_index
    metadata_list = []
    for original_index in dataset_split:
        metadata_list.append({
            'source_dataset': original_dataset_name,
            'original_index': int(original_index)
        })

    # Save metadata
    metadata_file = os.path.join(save_dir, f"{split_dataset_name}.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4)

    print(f"Finished saving {split_name} metadata. Saved in {metadata_file}")
    return metadata_list


def split_dataset_main(dataloader, dataset_name, random_seed):
    # load all data from the dataloader
    full_dataset_indices = []
    for batch in tqdm(dataloader, desc=f"Loading dataset '{dataset_name}'"):
        # we only need the indices
        full_dataset_indices.extend(batch['index'])

    # shuffle and split the indices
    random.seed(random_seed)
    random.shuffle(full_dataset_indices)

    split_point = len(full_dataset_indices) // 2
    split1_indices = full_dataset_indices[:split_point]
    split2_indices = full_dataset_indices[split_point:]

    # save the two new datasets (metadata only)
    save_split_metadata_only(
        dataset_split=split1_indices,
        original_dataset_name=dataset_name,
        split_name="split1"
    )

    save_split_metadata_only(
        dataset_split=split2_indices,
        original_dataset_name=dataset_name,
        split_name="split2"
    )


if __name__ == "__main__":
    DATASET_TO_SPLIT_CONFIG = dataset_config_2
    DATASET_OUTPUT_DIR = HOLD_OUT_OUTPUT_DIR

    tokenizer = CLIPTokenizer.from_pretrained(os.path.join(attack_args.base_model_id, "tokenizer"))

    dataloader_to_split, _ = create_dataset_dataloader(
        output_dir=DATASET_OUTPUT_DIR,
        dataset_config=DATASET_TO_SPLIT_CONFIG,
        tokenizer=tokenizer,
        img_resolution=224,
        need_save=False,
        batch_size=1,
    )

    split_dataset_main(
        dataloader=dataloader_to_split,
        dataset_name=DATASET_TO_SPLIT_CONFIG['dataset'],
        random_seed=attack_args.global_seed
    )
