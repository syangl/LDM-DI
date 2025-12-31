import fiftyone as fo
import fiftyone.zoo as foz

OUTPUT_DIR = "./dataset"
DATASET = "coco-2014"
SPLIT = "validation" # "train" or "validation"
num_samples_to_download = 20000


if __name__ == '__main__':
    print(f"Attempting to download and load {num_samples_to_download} random samples from {DATASET} {SPLIT} set with captions...")
    try:
        dataset = foz.load_zoo_dataset(
            DATASET,
            split=SPLIT,
            label_types=["captions"],
            dataset_dir=OUTPUT_DIR,
            max_samples=num_samples_to_download,
            shuffle=True,
            drop_existing_dataset=False,
            persistent=True,
            name=f"{DATASET}-{SPLIT}-{num_samples_to_download//1000}k-subset"
        )

        print(f"\nSuccessfully loaded {len(dataset)} samples into dataset: '{dataset.name}'")

        # Visualize the dataset in the FiftyOne App
        session = fo.launch_app(dataset)

    except Exception as e:
        print(f"An error occurred: {e}")

    print("\nDownloading complete.")
