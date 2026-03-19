import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from exp_utils.load_util.load_data import create_dataset_dataloader
from exp_utils.config_util.attack_config import attack_args, dataset_plot_names, TRAIN_OUTPUT_DIR, dataset_config_1, HOLD_OUT_OUTPUT_DIR, dataset_config_2, STATISTIC_ROOT_PATH
from transformers import CLIPTokenizer
import json


def check_dataset_complexity(dataloader1,
                             dataloader2,
                             DATASET1_NAME,
                             DATASET2_NAME,
                             analysis_output_dir,
                             mode='raw'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cal_metrics_batch(imgs):
        metrics = {}
        # Laplacian Variance
        # https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
        gray = imgs.mean(dim=1, keepdim=True)
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        edges = F.conv2d(gray, kernel, padding=1)
        metrics['laplacian'] = torch.var(edges, dim=[1, 2, 3]).cpu().numpy()

        # HF Proportion
        # FFT and shift
        f = torch.fft.fft2(gray.squeeze(1))
        fshift = torch.fft.fftshift(f)
        magnitude = torch.abs(fshift)
        # create ideal low-pass filter mask
        rows, cols = gray.shape[-2:]
        center_row, center_col = rows // 2, cols // 2
        r = int(min(rows, cols) * 0.1)
        y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
        mask = ((x.to(device) - center_col)**2 + (y.to(device) - center_row)**2) <= r**2
        # calculate proportion
        total_magnitude = torch.sum(magnitude, dim=[1, 2])
        low_component = torch.sum(magnitude * mask, dim=[1, 2])
        metrics['fft'] = ((total_magnitude - low_component) / (total_magnitude + 1e-8)).cpu().numpy()

        return metrics


    data_store = {
        "ds1": {'laplacian': [], 'fft': []},
        "ds2": {'laplacian': [], 'fft': []}
    }

    def scan_dataset(dataloader, key, desc_name):
        with torch.no_grad():
            for batch in tqdm(dataloader):
                imgs = batch['image'].to(device)
                batch_res = cal_metrics_batch(imgs)
                for m_key in batch_res:
                    data_store[key][m_key].append(batch_res[m_key])

    scan_dataset(dataloader1, "ds1", DATASET1_NAME)
    scan_dataset(dataloader2, "ds2", DATASET2_NAME)

    for main_key in data_store:
        for m_key in data_store[main_key]:
            data_store[main_key][m_key] = np.concatenate(data_store[main_key][m_key])

    metric_configs = [
        ('laplacian', 'Laplacian Var'),
        ('fft', 'High-Freq Energy')
    ]

    detailed_report = {}

    print("\n" + "=" * 80)
    print(f"{'Metric':<30} | {'Overlap':<10} | {'Mean Diff'}")
    print("-" * 80)

    # plot
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for idx, (m_key, title) in enumerate(metric_configs):
        val1 = data_store['ds1'][m_key]
        val2 = data_store['ds2'][m_key]

        # overlap
        min_v, max_v = min(val1.min(), val2.min()), max(val1.max(), val2.max())
        bins = np.linspace(min_v, max_v, 50)
        hist1, _ = np.histogram(val1, bins=bins, density=True)
        hist2, _ = np.histogram(val2, bins=bins, density=True)
        bin_width = bins[1] - bins[0]
        overlap_percent = np.sum(np.minimum(hist1, hist2)) * bin_width * 100


        # difference of means
        val1_mean = val1.mean()
        val2_mean = val2.mean()
        diff_percent = np.abs(val1_mean - val2_mean) / max(val1_mean, val2_mean)  * 100

        detailed_report[m_key] = {
            "name": title,
            "overlap_percent": float(overlap_percent),
            "mean_diff_percent": float(diff_percent)
        }

        print(f"{title:<30} | {overlap_percent:>6.1f}% | {diff_percent:+.1f}%")

        # plot each metric on its respective subplot
        ax = axes[idx]
        ax.hist(val1, bins=bins, alpha=0.5, density=True, label=DATASET1_NAME, color='red')
        ax.hist(val2, bins=bins, alpha=0.5, density=True, label=DATASET2_NAME, color='blue')
        ax.set_title(f"{title}\nOverlap: {overlap_percent:.1f}%")
        ax.set_xlabel("Value")
        if idx == 0:
            ax.set_ylabel("Density")
        ax.legend(prop={'size': 8})
        ax.tick_params(axis='both', which='major', direction='in', labelsize=20)
        ax.grid(True, alpha=0.2)

    print("-" * 80)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', direction='in', labelsize=20)

    save_path = os.path.join(analysis_output_dir, f"{DATASET1_NAME}_{DATASET2_NAME}_complexity_report.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return detailed_report


if __name__ == "__main__":
    tokenizer = tokenizer = CLIPTokenizer.from_pretrained(os.path.join(attack_args.base_model_id, "tokenizer"))
    dataloader1, _ = create_dataset_dataloader(TRAIN_OUTPUT_DIR,
                                               dataset_config_1,
                                               tokenizer,
                                               512,
                                               64,
                                               need_save=False)
    dataloader2, _ = create_dataset_dataloader(HOLD_OUT_OUTPUT_DIR,
                                               dataset_config_2,
                                               tokenizer,
                                               512,
                                               64,
                                               need_save=False)

    # get plot dataset names
    DATASET1_NAME = "Dataset 1"
    DATASET2_NAME = "Dataset 2"
    # get dataset 1 name from config
    for key, value in dataset_plot_names.items():
        if dataset_config_1['dataset'].startswith(key):
            DATASET1_NAME = value
            break
    # get dataset 2 name from config
    for key, value in dataset_plot_names.items():
        if dataset_config_2['dataset'].startswith(key):
            DATASET2_NAME = value
            break

    analysis_output_dir = os.path.join(STATISTIC_ROOT_PATH, "complex_analysis")
    os.makedirs(analysis_output_dir, exist_ok=True)
    detailed_report = check_dataset_complexity(dataloader1, dataloader2, DATASET1_NAME, DATASET2_NAME, analysis_output_dir)

    with open(os.path.join(analysis_output_dir, f"{DATASET1_NAME}_{DATASET2_NAME}_complexity_report.txt"), "w", encoding='utf-8') as f:
        f.write(json.dumps(detailed_report, indent=4))
