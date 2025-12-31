import json
import os
from tqdm import tqdm

# Paths
YEAR = "2014"
ANNOTATION_PATH = f"YOUR_LOCAL_DATASET/coco_{YEAR}_val/raw/captions_val{YEAR}.json"
IMAGE_DIR = f"YOUR_LOCAL_DATASET/coco_{YEAR}_val/data/val"
META_OUTPUT_DIR = f"YOUR_LOCAL_DATASET/coco_{YEAR}_val/metadata/meta_coco_{YEAR}_val.jsonl"

# Load annotations
with open(ANNOTATION_PATH, "r") as f:
    data = json.load(f)

# Get downloaded image names
downloaded_images = set(os.listdir(IMAGE_DIR))

# Build image_id_to_captions map
image_id_to_captions = {}
for ann in data["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"]
    if image_id not in image_id_to_captions:
        image_id_to_captions[image_id] = []
    image_id_to_captions[image_id].append(caption)

# Build metadata
metadata = []

# Filter downloaded images
for img in data["images"]:
    file_name = img["file_name"]
    if file_name in downloaded_images:
        image_id = img["id"]
        local_path = os.path.join(IMAGE_DIR, file_name)
        metadata.append({
            "index": int(image_id),
            "local_path": local_path,
            "caption": image_id_to_captions.get(image_id, []),
        })

# Rename image file name
if not os.path.exists(META_OUTPUT_DIR):
    for meta in tqdm(metadata, desc="Renaming image files", total=len(metadata)):
        file_name = meta["local_path"].split("/")[-1]
        new_file_name = f"{meta['index']}.{file_name.split('.')[-1]}"
        new_local_path = os.path.join(IMAGE_DIR, new_file_name)
        os.rename(meta["local_path"], new_local_path)
        meta["local_path"] = new_local_path

# Save metadata
os.makedirs(os.path.dirname(META_OUTPUT_DIR), exist_ok=True)
with open(META_OUTPUT_DIR, "w", encoding="utf-8") as f:
    for meta in metadata:
        f.write(json.dumps(meta) + "\n")

print(f"Generate metadata for {len(metadata)} images. Saved to path: {META_OUTPUT_DIR}")
