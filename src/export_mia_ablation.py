"""
Export standard and calibrated DI results from saved ablation runs.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='Matrix root containing model/case/experiments/MIA_ablation runs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    expected = {}
    for path in sorted(args.results.glob('*/*/experiments/MIA_ablation/*/protocol.json')):
        protocol = json.loads(path.read_text())
        model = protocol['model']
        expected[model] = {'full', *protocol['mia_columns']}
        standard_threshold = float(norm.ppf(1 - protocol['alpha']))
        for case in protocol['variants']:
            key = model, case
            if key in rows:
                raise ValueError(f'Duplicate model and ablation case: {key}')
            run = path.parent / case
            summary = json.loads((run / 'summary.json').read_text())
            evaluation = {}
            for pair in ('holdhold', 'sushold'):
                if summary['reference_ttest_dir'] is not None:
                    target = 'commoncatalog_cc_by_sa' if model == 'commoncanvas-s-c' else 'laion_aesthetics_v2_5p0'
                    filename = 'ttest_pair_D_hold_out_1_D_hold_out_2_results.json' if pair == 'holdhold' else f'ttest_pair_{target}_D_hold_out_{target}_results.json'
                    source = Path(summary['reference_ttest_dir']) / filename
                else:
                    source = run / 'ttest' / f'pair_{pair}' / f'ttest_pair_{pair}_results.json'
                records = json.loads(source.read_text())
                evaluation[pair] = np.array([record['t_statistic'] for record in records])
                if len(records) != protocol['evaluation_splits'] or not np.isfinite(evaluation[pair]).all():
                    raise ValueError(f'Invalid evaluation statistics: {run}/{pair}')
            false_positives = int(np.sum(evaluation['holdhold'] > summary['threshold']))
            true_positives = int(np.sum(evaluation['sushold'] > summary['threshold']))
            assert false_positives == summary['false_positives']
            assert true_positives == summary['true_positives']
            rows[key] = {
                'model': model, 'removed_mia': case,
                'mia_methods': summary['mia_methods'], 'feature_columns': summary['feature_columns'],
                'calibration_pairs': summary['calibration_pairs'],
                'standard_threshold': standard_threshold, 'calibrated_threshold': summary['threshold'],
                'standard_false_positives': int(np.sum(evaluation['holdhold'] > standard_threshold)),
                'calibrated_false_positives': false_positives, 'fpr_trials': len(evaluation['holdhold']),
                'standard_true_positives': int(np.sum(evaluation['sushold'] > standard_threshold)),
                'calibrated_true_positives': true_positives, 'tpr_trials': len(evaluation['sushold']),
            }
    if not rows:
        raise ValueError(f'No ablation runs found: {args.results}')
    for model, cases in expected.items():
        if {case for name, case in rows if name == model} != cases:
            raise ValueError(f'Incomplete ablation cases: {model}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=next(iter(rows.values())).keys())
        writer.writeheader()
        writer.writerows(rows.values())
    summary = {
        'cases': len(rows),
        'fpr_reduced_cases': sum(row['calibrated_false_positives'] < row['standard_false_positives'] for row in rows.values()),
        'fpr_unchanged_cases': sum(row['calibrated_false_positives'] == row['standard_false_positives'] for row in rows.values()),
        'fpr_increased_cases': sum(row['calibrated_false_positives'] > row['standard_false_positives'] for row in rows.values()),
        'detection_maintained_cases': sum(row['calibrated_true_positives'] == row['standard_true_positives'] for row in rows.values()),
        'models': {},
    }
    reductions = [100 * (row['standard_false_positives'] - row['calibrated_false_positives']) / row['fpr_trials'] for row in rows.values()]
    summary['fpr_reduction_percentage_points_range'] = [min(reductions), max(reductions)]
    for model in expected:
        model_rows = [row for (name, _), row in rows.items() if name == model]
        summary['models'][model] = {
            'cases': len(model_rows),
            'standard_fpr_range': [min(row['standard_false_positives'] / row['fpr_trials'] for row in model_rows), max(row['standard_false_positives'] / row['fpr_trials'] for row in model_rows)],
            'calibrated_fpr_range': [min(row['calibrated_false_positives'] / row['fpr_trials'] for row in model_rows), max(row['calibrated_false_positives'] / row['fpr_trials'] for row in model_rows)],
            'calibrated_detection_range': [min(row['calibrated_true_positives'] / row['tpr_trials'] for row in model_rows), max(row['calibrated_true_positives'] / row['tpr_trials'] for row in model_rows)],
        }
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Exported {len(rows)} cases to {args.output}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
