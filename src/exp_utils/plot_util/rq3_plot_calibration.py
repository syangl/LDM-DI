"""
Draw Figure 5 and export Table IV from saved Welch statistics.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb}'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 44
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['ytick.labelsize'] = 40
plt.rcParams['axes.titlesize'] = 48
plt.rcParams['legend.fontsize'] = 40
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['lines.linewidth'] = 3.5 

def load_t_stats_for_model(model_dir):
    """Load the saved calibration and evaluation statistics for one model."""
    base12_stats, hold_hold_stats, sus_hold_stats = [], [], []
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    pair_dirs = [d for d in os.listdir(model_dir) if d.startswith("pair_") and os.path.isdir(os.path.join(model_dir, d))]
    for pair_dir in sorted(pair_dirs):
        pair_idx = pair_dir.split("_")[1]
        file_path = os.path.join(model_dir, pair_dir, f"ttest_pair_{pair_idx}_results.json")
        if not os.path.exists(file_path): raise FileNotFoundError(file_path)
            
        with open(file_path, "r", encoding='utf-8') as f:
            results = json.load(f)
            for res in results:
                t_val = res['t_statistic']
                if pair_idx.isdigit(): base12_stats.append(t_val)
                elif pair_idx == "holdhold": hold_hold_stats.append(t_val)
                elif pair_idx == "sushold": sus_hold_stats.append(t_val)
                
    if not all([base12_stats, hold_hold_stats, sus_hold_stats]):
        raise ValueError(f"Missing calibration or evaluation statistics: {model_dir}")
    return np.array(base12_stats), np.array(hold_hold_stats), np.array(sus_hold_stats)

def calculate_new_metrics(base12_stats, hold_hold_stats, sus_hold_stats, target_alpha=0.001):

    tau_std = stats.norm.ppf(1 - target_alpha)
    tau_calibrated = np.percentile(base12_stats, 100 * (1 - target_alpha))
    
    # 2. Empirical FPR
    fpr_std = np.mean(hold_hold_stats > tau_std) if len(hold_hold_stats) > 0 else 0
    fpr_cal = np.mean(hold_hold_stats > tau_calibrated) if len(hold_hold_stats) > 0 else 0

    # 4. TPR at Specific Thresholds
    tpr_std = np.mean(sus_hold_stats > tau_std) if len(sus_hold_stats) > 0 else 0
    tpr_cal = np.mean(sus_hold_stats > tau_calibrated) if len(sus_hold_stats) > 0 else 0

    return {
        "tau_std": tau_std,
        "tau_cal": tau_calibrated,
        "fpr_std": fpr_std,
        "fpr_cal": fpr_cal,
        "tpr_std": tpr_std,
        "tpr_cal": tpr_cal,
    }

def print_latex_table(models_info, all_metrics, alpha):
    """Print the model metrics as a LaTeX table."""
    print("\n" + "="*80)
    print("RQ3: Compact Performance Comparison Table (LaTeX Code)")
    print("="*80)
    
    print(r"\begin{table}[!t]")
    print(r"\centering")
    print(r"\caption{Performance comparison under realistic similar distribution shift. Thresholds are calculated at significance level $\alpha=0.001$. TPR is evaluated at the specific threshold $\tau$. Best results are highlighted in bold.}")
    print(r"\label{tab:rq3}")
    
    print(r"\resizebox{\columnwidth}{!}{")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Method} & \textbf{Threshold $\tau$} & \textbf{Empirical FPR $\downarrow$} & \textbf{TPR @ $\tau \uparrow$} \\")
    print(r"\midrule")
    
    for i, model in enumerate(models_info):
        metrics = all_metrics[i]

        print(f"\\rowcolor{{gray!15}} \\multicolumn{{4}}{{c}}{{\\textbf{{\\textit{{Target Model: {model['name']}}}}}}} \\\\")
        print(r"\midrule")

        print(f"Standard DI & {metrics['tau_std']:.3f} & {metrics['fpr_std']*100:.2f}\\% & {metrics['tpr_std']*100:.2f}\\% \\\\")

        print(f"\\textbf{{Ours (Calibrated)}} & \\textbf{{{metrics['tau_cal']:.3f}}} & \\textbf{{{metrics['fpr_cal']*100:.2f}\\%}} & {metrics['tpr_cal']*100:.2f}\\% \\\\")

        if i < len(models_info) - 1:
            print(r"\midrule")
            
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table}")
    print("="*80 + "\n")

def save_csv_table(models_info, all_metrics, alpha, output_dir):
    """Save thresholds and rejection rates as CSV."""
    import csv
    target_pct = alpha * 100
    
    csv_path = os.path.join(output_dir, "RQ3_3Models_Results.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow([
            "Model", 
            "Method", 
            "Threshold",  
            "Empirical FPR (%)", 
            "TPR @ tau (%)", 
        ])
        
        for i, model in enumerate(models_info):
            metrics = all_metrics[i]

            writer.writerow([
                model['name'],
                "Standard DI",
                f"{metrics['tau_std']:.3f}",
                f"{metrics['fpr_std']*100:.2f}",
                f"{metrics['tpr_std']*100:.2f}",
            ])

            writer.writerow([
                model['name'],
                "Ours (Calibrated)",
                f"{metrics['tau_cal']:.3f}",
                f"{metrics['fpr_cal']*100:.2f}",
                f"{metrics['tpr_cal']*100:.2f}",
            ])
    
    print(f"CSV Results saved to: {csv_path}")

def plot_3models_kde(results_root, output_dir, alpha=0.001, k_base=400):
    models_info = [
        {"dir": os.path.join(results_root, "experiments/RQ3_calibration_performance/stable-diffusion-v1p5/ttest"), "name": "Stable-Diffusion-v1-5"},
        {"dir": os.path.join(results_root, "experiments/RQ3_calibration_performance/commoncanvas-s-c/ttest"), "name": "CommonCanvas-S-C"},
        {"dir": os.path.join(results_root, "experiments/RQ3_calibration_performance/kandinsky-3/ttest"), "name": "Kandinsky-3"}
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(32, 10))
    tau_std = stats.norm.ppf(1 - alpha) 

    all_metrics = []

    for idx, (ax, model) in enumerate(zip(axes, models_info)):
        base12_stats, hold_hold_stats, sus_hold_stats = load_t_stats_for_model(model["dir"])
        if k_base is not None:
            random.seed(0)
            base12_stats = np.array(random.sample(base12_stats.tolist(), k_base))

        metrics = calculate_new_metrics(base12_stats, hold_hold_stats, sus_hold_stats, target_alpha=alpha)
        all_metrics.append(metrics)
        tau_calibrated = metrics['tau_cal']

        sns.kdeplot(hold_hold_stats, ax=ax, color='gray', fill=True, alpha=0.4, linewidth=0, 
                    label=r'True Non-member ($D_{\text{hold}}$ vs. $D_{\text{hold}}$)')

        sns.kdeplot(sus_hold_stats, ax=ax, color='#1F77B4', fill=True, alpha=0.4, linewidth=0, 
                    label=r'True Member ($D_{\text{sus}}$ vs. $D_{\text{hold}}$)')

        ax.axvline(x=tau_std, color='black', linestyle=':', linewidth=4, 
                    label=f'Standard DI $\\tau_{{std}}$')
        ax.axvline(x=tau_calibrated, color='#D62728', linestyle='-', linewidth=4, 
                    label=f'Calibrated $\\tau_{{\\alpha}}$')

        subfig_labels = [f'(a)', '(b)', '(c)']
        ax.set_title(f"{subfig_labels[idx]} {model['name']}", y=-0.28, fontweight='bold', fontsize=44)
        ax.set_xlabel("Welch's $t$-statistic ($WT$)", labelpad=20)
        
        if idx == 0:
            ax.set_ylabel("Density", labelpad=20)
        else:
            ax.set_ylabel("")

            ax.tick_params(axis='y', labelleft=False)
        
        ax.tick_params(axis='both', which='major', length=10, width=2.5, direction='in')
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.5)
        
        x_min = np.min(hold_hold_stats) - 2 if len(hold_hold_stats) > 0 else -5
        x_max = np.max(sus_hold_stats) + 5 if len(sus_hold_stats) > 0 else 5
        ax.set_xlim(x_min, x_max)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 1.15), 
               ncol=4, frameon=False, columnspacing=1.5, handlelength=2.5)

    plt.tight_layout()


    pdf_path = os.path.join(output_dir, "RQ3_3Models_Distribution.pdf")
    png_path = os.path.join(output_dir, "RQ3_3Models_Distribution.png")
    plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    
    print(f"3-Model Plot saved to:\n- {pdf_path}\n- {png_path}")

    print_latex_table(models_info, all_metrics, alpha)

    save_csv_table(models_info, all_metrics, alpha, output_dir)
    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot RQ3 calibration distributions and detection rates")
    parser.add_argument('--results', required=True, help='Input root containing experiments')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=False)
    plot_3models_kde(args.results, args.output)
