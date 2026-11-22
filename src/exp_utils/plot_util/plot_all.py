"""
Generate RQ1, RQ2, RQ3, RQ5 and RQ6 figures and tables.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


def main(results, output):
    analysis = output / 'plot_analyze'
    analysis.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    start = time.perf_counter()
    for script, section in [('rq1_plot_vulnerability.py', 'RQ1_standard_di_vulnerability'),
                            ('rq3_plot_calibration.py', 'RQ3_calibration_performance')]:
        subprocess.run([sys.executable, str(scripts / script), '--results', str(results),
                        '--output', str(analysis / section)], check=True)
    for case, section in [('rq2', 'RQ2_hyperparameters'), ('rq5', 'RQ5_distribution_bound_ablation')]:
        subprocess.run([sys.executable, str(scripts / 'rq2_rq5_hyperparameter_analysis.py'),
                        '--results', str(results), '--output', str(output), '--case', case], check=True)
    rq3 = 'RQ3_calibration_performance'
    pd.testing.assert_frame_equal(
        pd.read_csv(analysis / rq3 / 'RQ3_3Models_Results.csv'),
        pd.read_csv(results / 'plot_analyze' / rq3 / 'RQ3_3Models_Results.csv'))
    rq6 = analysis / 'RQ6_dataset_size'
    rq6.mkdir()
    csv = rq6 / 'stable-diffusion-v1-5.csv'
    subprocess.run([sys.executable, str(scripts / 'rq6_analyze_dataset_sizes.py'),
                    '--results', str(results), '--output', str(csv)], check=True)
    saved = pd.read_csv(results / 'plot_analyze/RQ6_dataset_size/stable-diffusion-v1-5.csv')
    pd.testing.assert_frame_equal(pd.read_csv(csv), saved)
    subprocess.run([sys.executable, str(scripts / 'rq6_plot_dataset_sizes.py'), '--csv', str(csv),
                    '--output', str(rq6 / 'rq6_stable-diffusion-v1-5.pdf')], check=True)
    # Export the classifier validation table from saved accuracies.
    shared = analysis / 'shared_res'
    shared.mkdir()
    similarity = results / 'experiments/shared_res/classifier_validation'
    rows = []
    for file in sorted((similarity / 'semantic_bound').glob('*metrics.json')):
        with file.open() as f:
            rows.append({'pair': file.stem.removesuffix('_metrics'), 'ACC': json.load(f)['Clf_Acc']})
    pd.DataFrame(rows).to_csv(shared / 'table_II.csv', index=False)
    rows = []
    similarity = results / 'experiments/RQ2_hyperparameters/distribution_metrics'
    for group in ['5p0_hold', 'ccby_hold', 'Dhold12']:
        with next((similarity / group / 'semantic_bound').glob('*metrics.json')).open() as f:
            acc = json.load(f)['Clf_Acc']
        with next((similarity / group / 'texture_bound').glob('*.txt')).open() as f:
            texture = json.load(f)
        rows.append({'pair': group, 'delta_sem': 2 * abs(2 * acc - 1),
                     'delta_txt': max(1 - v['overlap_percent'] / 100 for v in texture.values())})
    pd.DataFrame(rows).to_csv(analysis / 'RQ2_hyperparameters/table_III.csv', index=False)
    with (analysis / 'plot_time.json').open('x') as f:
        json.dump({'operation': 'plot_all', 'elapsed_seconds': time.perf_counter() - start}, f, indent=2)
    print(f'Saved figures and tables to {analysis}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True, help='Input root containing experiments and plot_analyze')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[3] / 'exp_data',
                        help='Output root for figures, tables, and calibration pools')
    args = parser.parse_args()
    main(args.results.resolve(), args.output.resolve())
