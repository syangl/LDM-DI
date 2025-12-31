import os
import json
import random
import requests
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import io
import threading
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

counter_lock = threading.Lock()


def create_session(retries=3, backoff_factor=0.3):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def stream_filter_sample(dataset_name,
                         sample_size,
                         score_threshold,
                         split,
                         SHUFFLE_SEED):
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    print(f"Streaming dataset {dataset_name} ...")
    random.seed(SHUFFLE_SEED)
    reservoir = []
    total = 0
    progress_bar = tqdm(total=sample_size, desc="Sampling")
    for sample in dataset:
        try:
            score = float(sample["AESTHETIC_SCORE"])
        except Exception:
            continue
        if score >= score_threshold:
            total += 1
            progress_bar.update(1)
            if len(reservoir) < sample_size:
                reservoir.append(sample)
            else:
                idx = random.randint(0, total - 1)
                if idx < sample_size:
                    reservoir[idx] = sample
        # early stop if we collected enough examples
        if total >= sample_size * 5:  # if we have randomly enough replace examples enough, stop
            break
    print(f"Sampled example num: {len(reservoir)} / sum: {total}")
    return reservoir


def download_and_save(url, save_dir, index):
    session = create_session()
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))
        img = img.convert('RGB')
        img_path = os.path.join(save_dir, f"{index}.jpg")
        img.save(img_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"[Error {index}]: {str(e)}")
        return False


def save_samples(samples, save_dir, max_workers):
    os.makedirs(save_dir, exist_ok=True)
    tasks = []
    for sample in samples:
        idx = sample["index"]
        url = sample["image_url"]
        tasks.append((url, save_dir, idx))
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_and_save, *task): task
            for task in tasks
        }
        with tqdm(total=len(tasks), desc="Downloading images") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        with counter_lock:
                            success_count += 1
                except Exception as e:
                    print(f"Task error: {str(e)}")
                pbar.update(1)
    print(f"Successfully downloaded: {success_count}/{len(tasks)}")


def main():
    SHUFFLE_SEED = 256
    SPLIT = "train"
    SAMPLE_SAVE_SIZE = 4000
    SCORE_THRESHOLD = 5.0 # score threshold for aesthetics you want to download
    SAVE_DIR = os.path.join("./data", SPLIT)
    META_PATH = f"./meta_laion_aesthetics_v2_{str(SCORE_THRESHOLD).replace('.', 'p')}.jsonl"
    HF_REPO = "laion/aesthetics_v2_4.75"
    with open("./config.json", 'w', encoding='utf-8') as f:
        config = {
            "HF_REPO": HF_REPO,
            "SPLIT": SPLIT,
            "SCORE_THRESHOLD": SCORE_THRESHOLD,
            "SAMPLE_SAVE_SIZE": SAMPLE_SAVE_SIZE,
            "SAVE_DIR": SAVE_DIR,
            "SHUFFLE_SEED": SHUFFLE_SEED
        }
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Config:", config)

    if not os.path.exists(META_PATH):
        print("Starting to stream dataset ...")
        samples = stream_filter_sample(dataset_name=HF_REPO,
                                       sample_size=SAMPLE_SAVE_SIZE,
                                       score_threshold=SCORE_THRESHOLD,
                                       split=SPLIT,
                                       SHUFFLE_SEED=SHUFFLE_SEED)
        with open(META_PATH, 'w', encoding='utf-8') as f:
            for idx, sample in enumerate(samples):
                meta = {
                    "index": idx,
                    "caption": sample["TEXT"],
                    "aesthetic_score": sample["AESTHETIC_SCORE"],
                    "image_url": sample["URL"],
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"Metadata saved to {META_PATH}")

    # get metadata
    samples = []
    with open(META_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Start downloading to {SAVE_DIR} ...")
    save_samples(samples, SAVE_DIR, max_workers=20)
    print(f"All tasts finished, sum: {len(samples)}")


if __name__ == '__main__':
    main()
