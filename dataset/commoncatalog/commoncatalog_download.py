import os
import json
import random
import io
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image


def stream_and_save(dataset_name,
                    sample_start,
                    sample_size,
                    split,
                    save_dir,
                    meta_path,
                    random_seed=None,
                    max_samples_to_consider=100000,):
    os.makedirs(save_dir, exist_ok=True)
    
    if random_seed is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")

    if os.path.exists(dataset_name) and dataset_name.endswith('.parquet'):
        print(f"Loading local parquet file: {dataset_name}")
        dataset = load_dataset("parquet", data_files=dataset_name, split="train", streaming=True)
    else:
        print(f"Streaming dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    print(f"Will consider up to {max_samples_to_consider} samples for random selection")

    reservoir = []
    total_processed = 0
    considered_count = 0
    
    print("Selecting random samples using reservoir sampling...")
    for idx, sample in enumerate(dataset):
        total_processed += 1
        
        if idx < sample_start:
            continue
            
        considered_count += 1
        if considered_count > max_samples_to_consider:
            print(f"Reached max_samples_to_consider limit ({max_samples_to_consider}), stopping selection")
            break
            
        if len(reservoir) < sample_size:
            reservoir.append((idx, sample))
        else:
            j = random.randint(0, considered_count - 1)
            if j < sample_size:
                reservoir[j] = (idx, sample)
    
    print(f"Total samples processed: {total_processed}")
    print(f"Samples considered for selection: {considered_count}")
    print(f"Selected {len(reservoir)} random samples")

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        with tqdm(total=len(reservoir), desc="Saving selected samples") as pbar:
            for save_idx, (original_idx, sample) in enumerate(reservoir):
                # save image
                try:
                    jpg_data = sample["jpg"]
                    if isinstance(jpg_data, bytes):
                        img = Image.open(io.BytesIO(jpg_data)).convert("RGB")
                    else:
                        img = jpg_data.convert("RGB")
                    
                    img_path = os.path.join(save_dir, f"{save_idx}.jpg")
                    img.save(img_path, "JPEG", quality=95)

                    # metadata
                    meta = {
                        "index": save_idx,
                        "original_index": original_idx,
                        "caption": sample["blip2_caption"],
                        "image_url": sample["downloadurl"],
                    }
                    meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"[Error saving sample {original_idx}]: {e}")

                pbar.update(1)

    print(f"Finished: saved {len(reservoir)} random samples to {save_dir}")
    print(f"Metadata saved to {meta_path}")


def main():
    HF_REPO = "common-canvas/commoncatalog-cc-by-sa"

    SPLIT = "train" 
    SAMPLE_START = 0
    SAMPLE_SAVE_SIZE = 1000
    RANDOM_SEED = None  
    MAX_SAMPLES_TO_CONSIDER = 100000  

    SAVE_DIR = os.path.join("./data", SPLIT)
    META_PATH = f"./metadata/meta_commoncatalog_cc_by_sa.jsonl"
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

    with open("./config.json", 'w', encoding='utf-8') as f:
        config = {
            "HF_REPO": HF_REPO,
            "SPLIT": SPLIT,
            "SAMPLE_SAVE_SIZE": SAMPLE_SAVE_SIZE,
            "SAVE_DIR": SAVE_DIR,
            "RANDOM_SEED": RANDOM_SEED,
            "MAX_SAMPLES_TO_CONSIDER": MAX_SAMPLES_TO_CONSIDER,
        }
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Config:", config)

    stream_and_save(dataset_name=HF_REPO,
                    sample_start=SAMPLE_START,
                    sample_size=SAMPLE_SAVE_SIZE,
                    split=SPLIT,
                    save_dir=SAVE_DIR,
                    meta_path=META_PATH,
                    random_seed=RANDOM_SEED,
                    max_samples_to_consider=MAX_SAMPLES_TO_CONSIDER,)


if __name__ == "__main__":
    main()
