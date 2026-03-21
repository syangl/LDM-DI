import json
import os
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp_utils.config_util.random_provider import global_random
from exp_utils.config_util.attack_config import attack_args
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tfms
import torch


class ImageDS(Dataset):

    def __init__(
        self,
        output_dir,
        dataset_config,
        tokenizer,
        img_resolution,
        need_save=False,
    ):
        self.output_dir = output_dir
        self.dataset_config = dataset_config
        self.need_save = need_save
        self.tokenizer = tokenizer
        self.data = []
        self.img_resolution = img_resolution
        self.transform = tfms.Compose([
            tfms.ToTensor(),
            tfms.Normalize([0.5], [0.5]),
        ])
        # prepare data
        self._prepare_data(self.tokenizer)

        if self.need_save:
            # save dataset images to output directory
            self._save_dataset_images()

    def _get_caption_by_index(self, metadata, index):
        ret = ""
        for data in metadata:
            if data.get('index') == index:
                try:
                    ret = data.get('caption')
                except Exception as e:
                    # ret = data.get('text')
                    print(f"Error getting caption for index {index}: {e}")
                # speed up: remove data from metadata
                metadata.remove(data)
        # if caption is None, set to empty string
        if ret == None:
            ret = ""
        return ret
    def _load_dataset(self, DATASET_CONFIG):
        load_path = DATASET_CONFIG["load_path"]
        ds = dict()
        image_name_list = os.listdir(load_path)
        caption_path = os.path.join(
            load_path.replace(f"data/{DATASET_CONFIG['split']}", ""), f"metadata/meta_{DATASET_CONFIG['dataset']}.jsonl")
        
        with open(caption_path, 'r', encoding='utf-8') as meta_f:
            metadata = [json.loads(line.strip()) for line in meta_f]

        assert DATASET_CONFIG['load_num'] <= len(image_name_list)
        
        global_random.seed(attack_args.global_seed)
        image_name_list = global_random.sample(image_name_list, DATASET_CONFIG['load_num'])
        
        for i, image_name in tqdm(enumerate(image_name_list), desc="loading dataset...", total=len(image_name_list)):
            image_path = os.path.join(load_path, image_name)
            index = int(image_name.split(".")[0])
            caption = self._get_caption_by_index(metadata, index)
            
            if isinstance(caption, list):
                caption = [c for c in caption if c is not None]
                if caption:
                    caption = max(caption, key=len)
                else:
                    caption = ""
            
            ds[str(index)] = [caption, image_path]
        
        print(f"Load {len(ds)} images from {load_path}.")
        print()
        return ds

    def _load_ds(self, DATASET_CONFIG):
        return self._load_dataset(DATASET_CONFIG)


    def _prepare_data(self, tokenizer):
        # generate prompts directly instead of reading from file
        print("Load prompts...")
        ds = self._load_ds(self.dataset_config)
        for index, example in ds.items():
            caption = example[0]
            dataset_image_path = example[1]
            img = Image.open(dataset_image_path).convert("RGB").resize((self.img_resolution, self.img_resolution))

            self.data.append({
                "index": index,
                "prompt": caption,
                "input_ids": tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.squeeze(),
                "image": img,
                "image_path": dataset_image_path,
            })

    def _save_dataset_images(self):
        print("Dataset images not found, loading & saving...")
        for data in tqdm(self.data, desc="save dataset images..."):
            # load image
            img = data['image']
            index = data['index']
            subdir = os.path.join(self.output_dir, str(index))
            os.makedirs(subdir, exist_ok=True)
            img.save(os.path.join(subdir, "dataset.jpg"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        index = data['index']
        prompt = data['prompt']
        input_ids = data['input_ids']
        img = self.transform(data['image'])
        image_path = data['image_path']
        return {
            'index': index,
            'input_ids': input_ids,
            'image': img,
            'prompt': prompt,
            'image_path': image_path,
        }


def collate_fn(examples):
    # check if pixel_values are already tensors or need to be converted
    image = torch.stack([example["image"] for example in examples])
    image = image.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    indexes = [example["index"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    return {"image": image, "input_ids": input_ids, "index": indexes, "prompt": prompts}

def create_dataset_dataloader(output_dir, dataset_config, tokenizer, img_resolution, batch_size=1, need_save=False):
    """
        Create a DataLoader for image and prompt dataset
    """
    dataset = ImageDS(output_dir, dataset_config, tokenizer, img_resolution, need_save)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    return dataloader, dataset


def get_img_dataLoader(dataset):
    return DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
