import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.config_util.attack_config import attack_args, HOLD_OUT_OUTPUT_DIR, dataset_config_2

import json
import random
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer


def save_split_dataset(
    dataset_split,
    original_dataset_name,
    original_load_path,
    split_name,
    output_dir="split_results"
):
    print(f"Saving {split_name} with {len(dataset_split)} images...")

    # paths
    split_dataset_name = f"{original_dataset_name}_{split_name}"
    save_dir = os.path.join(current_dir, output_dir, split_dataset_name, "data/train")
    meta_dir = os.path.join(current_dir, output_dir, split_dataset_name, "metadata")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    original_meta_file = os.path.join(os.path.dirname(os.path.dirname(original_load_path)), 'metadata', f'meta_{original_dataset_name}.jsonl')

    assert os.path.exists(original_meta_file) == True, f"Original metadata file not found: {original_meta_file}"

    # load original metadata into a dictionary for quick access
    src_metadata = {}
    with open(original_meta_file, 'r', encoding='utf-8') as src_f:
        for line in src_f:
            data = json.loads(line)
            src_metadata[data['index']] = data

    # save new metadata and copy images
    new_meta_path = os.path.join(meta_dir, f"meta_{split_dataset_name}.jsonl")
    with open(new_meta_path, 'w', encoding='utf-8') as dst_f:
        for index in tqdm(dataset_split, desc=f"Processing {split_name}"):
            # save metadata
            metadata = src_metadata[int(index)]
            dst_f.write(json.dumps(metadata) + '\n')

            # copy image file
            original_img_path = os.path.join(original_load_path, f"{index}.jpg")
            new_img_path = os.path.join(save_dir, f"{index}.jpg")
            Image.open(original_img_path).save(new_img_path)

    print(f"Finished saving {split_name}. Saved in {os.path.join(current_dir, output_dir, split_dataset_name)}")


def split_dataset_main(dataloader, dataset_name, dataset_load_path, random_seed):
    # load all data from the dataloader
    full_dataset_indices = []
    for batch in tqdm(dataloader, desc=f"Loading dataset '{dataset_name}'"):
        # we only need the indices to perform the split and locate original files
        full_dataset_indices.extend(batch['index'])

    # shuffle and split the indices
    random.seed(random_seed)
    random.shuffle(full_dataset_indices)

    split_point = len(full_dataset_indices) // 2
    split1_indices = full_dataset_indices[:split_point]
    split2_indices = full_dataset_indices[split_point:]


    # save the two new datasets
    save_split_dataset(
        dataset_split=split1_indices,
        original_dataset_name=dataset_name,
        original_load_path=dataset_load_path,
        split_name="split1"
    )

    save_split_dataset(
        dataset_split=split2_indices,
        original_dataset_name=dataset_name,
        original_load_path=dataset_load_path,
        split_name="split2"
    )



if __name__ == "__main__":
    DATASET_TO_SPLIT_CONFIG = dataset_config_2
    DATASET_OUTPUT_DIR = HOLD_OUT_OUTPUT_DIR

    tokenizer = tokenizer = CLIPTokenizer.from_pretrained(os.path.join(attack_args.base_model_id, "tokenizer"))

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
        dataset_load_path=DATASET_TO_SPLIT_CONFIG['load_path'],
        random_seed=attack_args.global_seed
    )
