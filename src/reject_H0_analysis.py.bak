import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import numpy as np
from exp_utils.config_util.attack_config import attack_args
from scipy.stats import combine_pvalues

# list of -ln(p-value) from hold-hold settings (baseline noise)
baseline_noise_logs = [1, 1, 1, 1, 1] # example values
current_experiment_p_log = [1, 1, 1, 1, 1]

def reject_h0(current_experiment_p_log, output_dir, model_name):
    noise_p_values = np.exp(-np.array(baseline_noise_logs))
    _, p_noise_combined = combine_pvalues(noise_p_values, method='fisher')

    if p_noise_combined == 0.0:
        combined_noise_neg_ln = np.inf
    else:
        combined_noise_neg_ln = -np.log(p_noise_combined)

    if np.isinf(combined_noise_neg_ln):
        threshold = np.inf
    else:
        threshold = combined_noise_neg_ln * (1 + 0.5)  # 50% safety margin

    exp_p_values = np.exp(-np.array(current_experiment_p_log))

    _, p_exp_combined = combine_pvalues(exp_p_values, method='fisher')

    if p_exp_combined == 0.0:
        combined_exp_neg_ln = np.inf
    else:
        combined_exp_neg_ln = -np.log(p_exp_combined)


    if np.isinf(threshold):
        is_significant = False
        gap = 0
    elif np.isinf(combined_exp_neg_ln):
        # if exp is infinite and threshold is finite
        is_significant = True
        gap = np.inf
    else:
        is_significant = combined_exp_neg_ln > threshold
        gap = combined_exp_neg_ln - threshold

    with open(os.path.join(output_dir, "significance_report.txt"), "w") as f:
        f.write("=" * 60 + "\n")
        f.write(" Detection Significance Verification\n")
        f.write("=" * 60 + "\n")
        f.write(f"Metric: Negative Natural Logarithm (-ln p)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline Noise (Fisher's Combined)\n")
        f.write(f"Noise Samples: {len(baseline_noise_logs)}\n")
        f.write(f"Combined Noise Level: {combined_noise_neg_ln:.2f}\n")
        f.write(f"Dynamic Threshold (Noise * (1 + 50%)): {threshold}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Experiment Result (Fisher's Combined)\n")
        f.write(f"Target Model: {model_name}\n")
        f.write(f"Individual Scores: {[f'{x:.2f}' for x in current_experiment_p_log]}\n")
        f.write(f"Combined Score: {combined_exp_neg_ln}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Conclusion\n")
        if is_significant:
            f.write(f"Result: Significant\n")
            f.write(f"Combined Score exceeds Threshold by {gap} (ln scale).\n")
            f.write(f"Reject Null Hypothesis.\n")
        else:
            f.write(f"Result: Not significant\n")
            f.write(f"Combined Score is below Threshold.\n")
            f.write(f"Cannot Reject Null Hypothesis.\n")
        f.write("=" * 60 + "\n")

if __name__ == "__main__":
    reject_h0(current_experiment_p_log=current_experiment_p_log,
                                      model_name=attack_args.model_name,
                                      output_dir="./")
