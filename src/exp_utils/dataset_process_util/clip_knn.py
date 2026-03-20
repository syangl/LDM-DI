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
from exp_utils.config_util.attack_config import attack_args, TRAIN_OUTPUT_DIR, dataset_config_1, HOLD_OUT_OUTPUT_DIR, dataset_config_2
import json
import clip
from transformers import CLIPTokenizer



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

    def extract_features(self, image_tensors, batch_size=64):
        features = []

        with torch.no_grad():
            for i in tqdm(range(0, len(image_tensors), batch_size), desc="Extracting features"):
                batch = image_tensors[i:i + batch_size]
                batch = torch.stack(batch).to(self.device)
                # extract feature
                feats = self.model.encode_image(batch)
                # normalize
                feats = feats / feats.norm(dim=-1, keepdim=True)
                features.append(feats.cpu().numpy())

        return np.vstack(features)

    def find_overlap_region_density(self, features1, features2, target_num):
        """
        Given features from dataset1, find the top target_num most similar examples in dataset2.
        """
        k = 5

        # cosine similarity
        # features1 shape: (N1, D), features2 shape: (N2, D)
        # sims shape: (N1, N2)
        sims = features1 @ features2.T

        # for each image in dataset2, find its k-nearest neighbors in dataset1
        sims21 = np.sort(sims.T, axis=1)[:, -k:]

        # the overlap score for each image in dataset2 is its average similarity to its KNN in dataset1
        overlap_score2 = sims21.mean(axis=1)

        # get top target_num examples from dataset2
        idx2 = np.argsort(-overlap_score2)[:target_num]

        mask2 = np.zeros_like(overlap_score2, dtype=bool)
        mask2[idx2] = True
        return mask2

    def find_max_overlap_examples(self, dataset1, dataset2, target_num):
        # get indices and images
        img_tensors1 = [item[1] for item in dataset1]
        indices2 = [item[0] for item in dataset2]
        img_tensors2 = [item[1] for item in dataset2]

        # extract features
        features1 = self.extract_features(img_tensors1)
        features2 = self.extract_features(img_tensors2)

        # get overlap masks for dataset2
        overlap_mask2 = self.find_overlap_region_density(features1, features2, target_num)

        # get overlap datasets
        overlap_dataset2 = [(indices2[i], img_tensors2[i]) for i in range(len(dataset2)) if overlap_mask2[i]]

        # ensure we have exactly target_num examples
        if len(overlap_dataset2) > target_num:
            selected_indices = np.random.choice(len(overlap_dataset2), target_num, replace=False)
            overlap_dataset2 = [overlap_dataset2[i] for i in selected_indices]

        return overlap_dataset2

    def save_overlap_images(self,
                            overlap_dataset_indices,
                            dataset1_name,
                            dataset2_name,
                            dataset_load_path,
                            sub_folder,
                            output_dir="overlap_results",
                            ):
        save_dir = os.path.join(current_dir, output_dir, f"{dataset2_name}_{dataset1_name}", f"data/{sub_folder}")
        os.makedirs(save_dir, exist_ok=True)
        meta_path = os.path.join(save_dir.replace(f"data/{sub_folder}", ""), "metadata")
        os.makedirs(meta_path, exist_ok=True)
        src_meta_file = dataset_load_path.replace(f'data/{sub_folder}', f'metadata/meta_{dataset2_name}.jsonl')

        with open(os.path.join(meta_path, f"meta_{dataset2_name}_{dataset1_name}.jsonl"), 'w',encoding='utf-8') as dst_f:
            with open(src_meta_file, 'r', encoding='utf-8') as src_f:
                src_metadata = {}
                for line in src_f:
                    line = json.loads(line)
                    src_metadata[line['index']] = line
                for index in overlap_dataset_indices:
                    # save images
                    img_pil = Image.open(
                        os.path.join(dataset_load_path, f"{index}.jpg"))
                    filepath = os.path.join(save_dir, f"{index}.jpg")
                    img_pil.save(filepath)
                    # save metadata
                    metadata = src_metadata[int(index)]
                    dst_f.write(json.dumps(metadata) + '\n')


def find_distribution_overlap(dataloader1, dataloader2, overlap_finder, target_num):
    """
    Find examples in dataloader2 with maximum distribution overlap with dataloader1.
    """
    dataset1 = []
    dataset2 = []

    # Process dataloaders
    for batch in tqdm(dataloader1, desc="Loading dataset 1"):
        images = batch['image']
        indexes = batch['index']
        for i in range(len(images)):
            img_tensor = images[i]
            dataset1.append((indexes[i], img_tensor))

    for batch in tqdm(dataloader2, desc="Loading dataset 2"):
        images = batch['image']
        indexes = batch['index']
        for i in range(len(images)):
            img_tensor = images[i]
            dataset2.append((indexes[i], img_tensor))

    print(f"Loaded {len(dataset1)} images for query dataset")
    print(f"Loaded {len(dataset2)} images for search dataset")

    overlap_dataset2 = overlap_finder.find_max_overlap_examples(dataset1, dataset2, target_num=target_num)

    print(f"Found {len(overlap_dataset2)} overlap examples in search dataset")

    return overlap_dataset2


def find_overlap_main(
    dataloader1,
    dataloader2,
    dataset1_name,
    dataset2_name,
    dataset2_load_path,
    target_num,
    split,
    output_dir="overlap_results",
):
    # Find overlapping examples
    overlap_finder = DistributionOverlapFinder()
    overlap_holdout = find_distribution_overlap(dataloader1, dataloader2, overlap_finder, target_num)

    overlap_holdout_indices = [idx for idx, _ in overlap_holdout]

    overlap_finder.save_overlap_images(
        overlap_dataset_indices=overlap_holdout_indices,
        dataset1_name=dataset1_name,
        dataset2_name=dataset2_name,
        dataset_load_path=dataset2_load_path,
        sub_folder=split,
        output_dir=output_dir,

    )

    print("Overlap images saved")


if __name__ == "__main__":
    TARGET_NUM = attack_args.dataset_total_num
    tokenizer = tokenizer = CLIPTokenizer.from_pretrained(os.path.join(attack_args.base_model_id, "tokenizer"))

    # dataset 1
    dataloader1, _ = create_dataset_dataloader(TRAIN_OUTPUT_DIR,
                                               dataset_config_1,
                                               tokenizer,
                                               224, 
                                               64,
                                               need_save=False)
    # dataset 2
    dataset_config_2['load_num'] = 3993  # load full dataset to find overlap
    dataloader2, _ = create_dataset_dataloader(HOLD_OUT_OUTPUT_DIR,
                                               dataset_config_2,
                                               tokenizer,
                                               224,
                                               64,
                                               need_save=False)

    # find TARGET_NUM images in dataset 2 that are most similar to dataset 1
    find_overlap_main(dataloader1=dataloader1,
                      dataloader2=dataloader2,
                      dataset1_name=dataset_config_1['dataset'],
                      dataset2_name=dataset_config_2['dataset'],
                      dataset2_load_path=dataset_config_2['load_path'],
                      target_num=TARGET_NUM,
                      split=dataset_config_2['split'],
                      )
