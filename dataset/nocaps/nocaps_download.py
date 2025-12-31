import os
import json
from datasets import load_dataset
from tqdm import tqdm


def stream_and_save(
    dataset_name,
    sample_start,
    sample_size,
    split,
    save_dir,
    meta_path,
):
    os.makedirs(save_dir, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    print(f"Streaming dataset {dataset_name} ...")

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        with tqdm(total=sample_size, desc="Streaming & Saving") as pbar:
            for idx, sample in enumerate(dataset):
                if idx < sample_start:
                    continue
                if idx >= sample_start + sample_size:
                    break

                # save image
                try:
                    img = sample["image"].convert("RGB")
                    img_path = os.path.join(save_dir, f"{idx}.jpg")
                    img.save(img_path, "JPEG", quality=95)

                    # metadata
                    meta = {
                        "index": idx,
                        "caption": max(sample["annotations_captions"], key=len),
                        "image_url": sample["image_coco_url"],
                    }
                    meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"[Error {idx}]: {e}")

                pbar.update(1)

    print(f"Finished: saved {sample_size} samples to {save_dir}")
    print(f"Metadata saved to {meta_path}")


def main():
    HF_REPO = "HuggingFaceM4/NoCaps"
    SPLIT = "validation"
    SAMPLE_START = 0
    SAMPLE_SAVE_SIZE = 4500

    SAVE_DIR = os.path.join("./data", SPLIT)
    META_PATH = f"./metadata/meta_nocaps.jsonl"
    os.makedirs(os.path.dirname(META_PATH), exist_ok=True)

    with open("./config.json", 'w', encoding='utf-8') as f:
        config = {
            "HF_REPO": HF_REPO,
            "SPLIT": SPLIT,
            "SAMPLE_SAVE_SIZE": SAMPLE_SAVE_SIZE,
            "SAVE_DIR": SAVE_DIR,
        }
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Config:", config)

    stream_and_save(
        dataset_name=HF_REPO,
        sample_start=SAMPLE_START,
        sample_size=SAMPLE_SAVE_SIZE,
        split=SPLIT,
        save_dir=SAVE_DIR,
        meta_path=META_PATH,
    )


if __name__ == "__main__":
    main()
