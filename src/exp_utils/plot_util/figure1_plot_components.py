import os
from pathlib import Path
import argparse

import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser(description='Plot Figure 1 distribution components from saved MIA scores')
parser.add_argument('--results', type=Path, required=True, help='Directory containing Figure 1 score inputs under shift/inputs')
parser.add_argument('--output', type=Path, required=True, help='New output directory')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
import time

def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def get_example_data():

    train_1_path = str(args.results / 'shift/inputs' / 'mia-shift/com_cc_coco17/output/commoncatalog_cc_by/commoncanvas-s-c/rediffusemi/rediffusemi_member_scores.json')
    holdout_1_path = str(args.results / 'shift/inputs' / 'mia-shift/com_cc_coco17/output/coco_2017_val/commoncanvas-s-c/rediffusemi/rediffusemi_nonmember_scores.json')
    holdout_2_path = str(args.results / 'shift/inputs' / 'mia-shift/com_cc_multi/output/laion2b_multi_translated/commoncanvas-s-c/rediffusemi/rediffusemi_nonmember_scores.json')
    SAMPLE_SIZE = 2000

    seed1 = int(time.time()) % 10000
    print(seed1)
    np.random.seed(seed1)

    with open(train_1_path, 'r', encoding='utf-8') as f:
        loss_train_1 = json.load(f)['member_scores']
        loss_train_1 = np.random.choice(loss_train_1,
                                        size=SAMPLE_SIZE,
                                        replace=False)
    with open(holdout_1_path, 'r', encoding='utf-8') as f:
        loss_test_iid_1 = json.load(f)['nonmember_scores']
        loss_test_iid_1 = np.random.choice(loss_test_iid_1,
                                           size=SAMPLE_SIZE,
                                           replace=False)
    with open(holdout_2_path, 'r', encoding='utf-8') as f:
        loss_test_shifted_1 = json.load(f)['nonmember_scores']
        loss_test_shifted_1 = np.random.choice(loss_test_shifted_1,
                                               size=SAMPLE_SIZE,
                                               replace=False)

    train_2_path = str(args.results / 'shift/inputs' / 'mia-shift/sd_5p0_coco2017/output/laion_aesthetics_v2_5p0/stable-diffusion-v1p5/rediffusemi/rediffusemi_member_scores.json')
    holdout_2_1_path = str(args.results / 'shift/inputs' / 'mia-shift/sd_5p0_nocaps/output/nocaps/stable-diffusion-v1p5/rediffusemi/rediffusemi_nonmember_scores.json')
    holdout_2_2_path = str(args.results / 'shift/inputs' / 'mia-shift/sd_5p0_coco2017/output/coco_2017_val/stable-diffusion-v1p5/rediffusemi/rediffusemi_nonmember_scores.json')

    seed2 = int(time.time()) % 10000
    print(seed2)
    np.random.seed(seed2)

    with open(train_2_path, 'r', encoding='utf-8') as f:
        loss_train_2 = json.load(f)['member_scores']
        loss_train_2 = np.random.choice(loss_train_2,
                                        size=SAMPLE_SIZE,
                                        replace=False)
    with open(holdout_2_1_path, 'r', encoding='utf-8') as f:
        loss_test_iid_2 = json.load(f)['nonmember_scores']
        loss_test_iid_2 = np.random.choice(loss_test_iid_2,
                                        size=SAMPLE_SIZE,
                                        replace=False)
    with open(holdout_2_2_path, 'r', encoding='utf-8') as f:
        loss_test_shifted_2 = json.load(f)['nonmember_scores']
        loss_test_shifted_2 = np.random.choice(loss_test_shifted_2,
                                            size=SAMPLE_SIZE,
                                            replace=False)
    return (loss_train_1, loss_test_iid_1, loss_test_shifted_1,
            loss_train_2, loss_test_iid_2, loss_test_shifted_2)

def plot_clean_distribution_subplot(ax, loss_train, loss_test_iid,
                                    loss_test_shifted, title):
    """
    Draw score densities and mean lines for one model.
    """

    sns.set_style("ticks")

    # Since we want members to have smaller values, we can invert the loss values by subtracting them from a constant (e.g., max value in all datasets)
    all_values = np.concatenate([loss_train, loss_test_iid, loss_test_shifted])
    max_val = np.max(all_values)

    # Invert values so members have smaller values
    loss_train_inverted = max_val - loss_train
    loss_test_iid_inverted = max_val - loss_test_iid
    loss_test_shifted_inverted = max_val - loss_test_shifted

    mu_train = np.mean(loss_train_inverted)
    mu_iid = np.mean(loss_test_iid_inverted)
    mu_shift = np.mean(loss_test_shifted_inverted)

    sns.kdeplot(loss_train_inverted,
                fill=True,
                label='Members (Train)',
                color='#1f77b4',
                alpha=0.4,
                linewidth=2.5,
                ax=ax)
    sns.kdeplot(loss_test_iid_inverted,
                fill=True,
                label='Non-Members (IID Test)',
                color='#ff7f0e',
                alpha=0.4,
                linewidth=2.5,
                ax=ax)
    sns.kdeplot(loss_test_shifted_inverted,
                fill=True,
                label='Non-Members (Shifted/OOD)',
                color='#d62728',
                alpha=0.4,
                linewidth=2.5,
                ax=ax)

    ax.grid(False)

    ax.set_xlim(-0.1, 1.1)

    y_max = ax.get_ylim()[1]

    ax.vlines(
        [mu_train, mu_iid, mu_shift],
        0,
        y_max * 3.5,
        linestyles='--',
        colors=['#1f77b4', '#ff7f0e', '#d62728'],
        alpha=0.8,
        linewidth=3)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Normalized Loss Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)

    return y_max

def plot_clean_combined_distribution():
    """
    Draw the two distribution-shift panels from saved MIA scores.
    """

    loss_train_1, loss_test_iid_1, loss_test_shifted_1, loss_train_2, loss_test_iid_2, loss_test_shifted_2 = get_example_data(
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    y_max_1 = plot_clean_distribution_subplot(
        ax1, 
        loss_train_1, 
        loss_test_iid_1, 
        loss_test_shifted_1,
        'Impact of Distribution Shift on Structural MI')

    y_max_2 = plot_clean_distribution_subplot(
        ax2, 
        loss_train_2, 
        loss_test_iid_2, 
        loss_test_shifted_2,
        'Impact of Distribution Shift on SECMI')

    common_y_max = max(y_max_1, y_max_2)
    ax1.set_ylim(0, common_y_max)
    ax2.set_ylim(0, common_y_max)

    # ax1.tick_params(direction='in')
    # ax2.tick_params(direction='in')

    ax2.tick_params(labelleft=False)

    plt.tight_layout()

    plt.subplots_adjust(wspace=0.01)

    output_path = str(args.output)
    os.makedirs(output_path, exist_ok=True)

    save_path = os.path.join(
        output_path, 'distribution_shift_diff_combined_clean.png')
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    print(f"Distribution plot saved to: {save_path}")

if __name__ == "__main__":

    plot_clean_combined_distribution()
