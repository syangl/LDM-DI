"""
Run full-feature and leave-one-MIA-out DI.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exp_utils.config_util.attack_config import attack_args, OUTPUT_ROOT_PATH
from exp_utils.dataset_process_util.meta_feature_manager import create_loader
from main_ttest import process_single_pair
from calculate_threshold_fpr import empirical_quantile_mean, calculate_rejection_rate, sample_calibration_pairs


# Column order follows extract_features_for_dataset in extract_features.py.
MIA_COLUMNS = {
    'secmi': [0],
    'naivelossmi': [1],
    'piami': [2],
    'structuralmi': [3],
    'rediffusemi': [4],
    'gradient_maskingmi': [5, 6, 7, 8, 9],
    'multiple_lossmi': [10, 11, 12, 13, 14],
    'noise_optimmi': [15, 16],
}


def load_main_statistics(directory, selected, target):
    records = json.loads((directory / 'ttest_pair_D_base_1_D_base_2_results.json').read_text())
    calibration = {int(row['pair_idx']): row['t_statistic'] for row in records}
    null_stats = [calibration[pair] for pair in selected]
    evaluation = {}
    for pair, filename in [('holdhold', 'ttest_pair_D_hold_out_1_D_hold_out_2_results.json'),
                           ('sushold', f'ttest_pair_{target}_D_hold_out_{target}_results.json')]:
        records = json.loads((directory / filename).read_text())
        evaluation[pair] = np.array([row['t_statistic'] for row in records])
    return null_stats, evaluation


def main():
    torch.set_num_threads(1)
    args = attack_args
    run_dir = Path(OUTPUT_ROOT_PATH) / 'experiments/MIA_ablation' / args.model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    target = 'commoncatalog_cc_by_sa' if args.model_name == 'commoncanvas-s-c' else 'laion_aesthetics_v2_5p0'
    loader = create_loader()
    for source in [target, 'coco_2017_val', 'laion2b_multi_translated', 'nocaps']:
        values = loader.load_feature_dict(source)
        features = torch.stack(list(values.values()))
        if features.shape[1] != 17 or not torch.isfinite(features).all():
            raise ValueError(f'Expected finite 17-column features: {source}')

    candidates = [f'pair_{i}' for i in range(1, args.num_pairs + 1)]
    selected = [int(name.removeprefix('pair_')) for name in sample_calibration_pairs(candidates, args.k_base)]
    main_results = Path(args.main_results).resolve() if args.main_results else Path(OUTPUT_ROOT_PATH) / 'experiments/RQ3_calibration_performance'
    variants = [args.mia_variant] if args.mia_variant is not None else ['full', *MIA_COLUMNS]
    protocol = {
        'model': args.model_name, 'metadata_root': args.metadata_root,
        'feature_root': args.feature_root, 'local_dataset_path': args.local_dataset_path,
        'dataset_total_num': args.dataset_total_num, 'evaluation_splits': args.split_times,
        'candidate_pairs': args.num_pairs, 'calibration_pair_ids': selected,
        'global_seed': args.global_seed, 'main_results': str(main_results),
        'alpha': 0.001, 'quantile_method': 'numpy.percentile, linear',
        'mia_columns': MIA_COLUMNS, 'epochs': 1000, 'optimizer': 'Adam, lr=0.01',
        'torch_threads': 1, 'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'scope': 'Original full-feature main results with retrained, recalibrated leave-one-out variants. Original calibration pairs and evaluation splits are retained.',
        'record_order': 'calibration_pair_ids, holdhold splits, sushold splits',
        'prediction_groups': ['sus', 'hold_out'],
    }
    for removed in variants:
        output = run_dir / f'{removed}.npz'
        if output.exists():
            with np.load(output, allow_pickle=False) as saved:
                if json.loads(str(saved['metadata']))['protocol'] != protocol:
                    raise ValueError(f'Existing ablation inputs differ; use a new output_root: {output}')
            print(f'Ablation case is already complete: {output}', flush=True)
            continue
        start = time.perf_counter()
        columns = [i for i in range(17) if removed == 'full' or i not in MIA_COLUMNS[removed]]
        all_records = []
        reference_dir = main_results / args.model_name / 'ttest'
        if removed == 'full':
            calibration, evaluation = load_main_statistics(reference_dir, selected, target)
        else:
            calibration = []
            for number, pair in enumerate(selected, 1):
                records, _ = process_single_pair(
                    loader, pair, 'D_base_1', 'D_base_2', None, None,
                    1, args.dataset_total_num // 2, feature_columns=columns, seed=args.global_seed)
                assert len(records) == 1
                calibration.append(records[0]['t_statistic'])
                all_records.extend(records)
                if number % 50 == 0:
                    print(f'{removed}: calibration {number}/{args.k_base}', flush=True)

            evaluation = {}
            for pair, names in [
                ('holdhold', ('D_hold_out_1', 'D_hold_out_2')),
                ('sushold', (target, 'D_hold_out_' + target)),
            ]:
                records, _ = process_single_pair(
                    loader, pair, *names, None, None, args.split_times,
                    args.dataset_total_num // 2, feature_columns=columns, seed=args.global_seed)
                assert len(records) == args.split_times
                evaluation[pair] = np.array([record['t_statistic'] for record in records])
                all_records.extend(records)
        if any(len(values) != args.split_times for values in evaluation.values()):
            raise ValueError(f'Evaluation split count differs from the main experiment: {removed}')
        if not all(np.isfinite(values).all() for values in [calibration, *evaluation.values()]):
            raise ValueError(f'Non-finite test statistics: {removed}')
        tau = empirical_quantile_mean(calibration, alpha=0.001)['tau_quantile']
        fpr, false_positives = calculate_rejection_rate(evaluation['holdhold'], tau)
        tpr, true_positives = calculate_rejection_rate(evaluation['sushold'], tau)
        row = {
            'removed_mia': removed, 'mia_methods': 8 if removed == 'full' else 7,
            'feature_columns': len(columns), 'kept_columns': columns,
            'calibration_pairs': len(calibration), 'threshold': tau,
            'reference_ttest_dir': str(reference_dir) if removed == 'full' else None,
            'false_positives': int(false_positives), 'fpr_trials': args.split_times, 'fpr': float(fpr),
            'true_positives': int(true_positives), 'tpr_trials': args.split_times, 'tpr': float(tpr),
            'elapsed_seconds': time.perf_counter() - start,
        }
        arrays = {'calibration': np.asarray(calibration), **evaluation,
                  'metadata': np.array(json.dumps({'protocol': protocol, 'summary': row}))}
        if all_records:
            arrays.update(predictions=np.stack([r['predictions'] for r in all_records]),
                          weights=np.stack([r['weights'] for r in all_records]),
                          bias=np.array([r['bias'] for r in all_records]),
                          fit_seconds=np.array([r['test_elapsed_seconds'] for r in all_records]))
        pending = output.with_suffix('.partial.npz')
        np.savez_compressed(pending, **arrays)
        pending.replace(output)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
