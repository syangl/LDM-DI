"""
This script is used to clean bad images from downloaded dataset. 
Since some images of dataset urls are not valid as time goes on, we need to clean them.
"""
from cleanvision.imagelab import Imagelab
from pathlib import Path
import shutil
import os
from PIL import Image

def clean_bad_images():
    # set directory to be cleaned
    SOURCE_DIR = 'YOUR_LOCAL_DATASET/laion2b_multi_translated/data/val'

    # directory for storing all identified problematic images
    BAD_IMAGES_DIR = 'YOUR_LOCAL_DATASET/laion2b_multi_translated/data/bad_images_from_download'

    filepaths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    problematic_filepaths = []

    # check image size
    max_pixels = 6000 * 6000  # max pixel limit

    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(valid_extensions):
                filepath = os.path.join(root, file)
                try:
                    # skip too large images
                    with Image.open(filepath) as img:
                        # check large images
                        width, height = img.size
                        # # check truncated or corrupted images
                        # img.convert("RGB").resize((512, 512))
                        if width * height > max_pixels:
                            print(f"Skipping large image: {filepath} (size: {width}x{height})")
                            continue
                    filepaths.append(filepath)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    problematic_filepaths.append(filepath)
                    continue

    if not filepaths:
        print("No image files found")
        return

    print(f"Found {len(filepaths)} images to process")

    # create an Imagelab instance
    imagelab = Imagelab(filepaths=filepaths)
    # define the types of issues you want to detect
    issue_types = {
        'blurry': {},  # blurry images
        'dark': {},  # overly dark images
        'light': {'threshold': 0.2},  # overly bright images
        'near_duplicates': {},  # similar content duplicate images
        'exact_duplicates': {},  # identical content images
        'low_information': {},  # images with too little information
    }
    # run the issue finder
    print('Running issue finder...')
    imagelab.find_issues(issue_types=issue_types)
    # view summary of issue report
    print('Issues found:')
    issue_summary = imagelab.issues
    issue_summary.to_csv('issue_summary.csv')
    # get file paths of all images flagged as problematic
    issue_columns = [
        col for col in issue_summary.columns
        if col.startswith('is_') and col.endswith('_issue')
    ]
    problematic_filepaths.extend(issue_summary[issue_summary[issue_columns].any(axis=1)].index.tolist())

    if problematic_filepaths:
        bad_path = Path(BAD_IMAGES_DIR)
        bad_path.mkdir(exist_ok=True)

        # move all problematic images
        moved_count = 0
        for filepath_str in problematic_filepaths:
            filepath = Path(filepath_str)
            destination = bad_path / filepath.name
            shutil.move(str(filepath), str(destination))
            moved_count += 1

        print(f"Moved {moved_count} problematic images to {bad_path}")


if __name__ == "__main__":
    clean_bad_images()
