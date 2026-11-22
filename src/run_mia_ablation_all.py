from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

from exp_utils.config_util.attack_config import attack_args, OUTPUT_ROOT_PATH


def main():
    args = attack_args
    output = Path(OUTPUT_ROOT_PATH)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from run_mia_ablation import MIA_COLUMNS

    if args.feature_root is None or args.metadata_root is None:
        raise ValueError('Set feature_root and metadata_root to the saved experiment inputs')
    feature_root = os.path.abspath(args.feature_root)
    metadata_root = os.path.abspath(args.metadata_root)
    local_dataset_path = os.path.abspath(args.local_dataset_path)
    main_results = str(Path(args.main_results).resolve()) if args.main_results else str(output / 'experiments/RQ3_calibration_performance')
    models = args.models or ['stable-diffusion-v1p5', 'commoncanvas-s-c', 'kandinsky-3']
    scripts = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', 
               TQDM_DISABLE='1',
               OMP_NUM_THREADS='1', 
               MKL_NUM_THREADS='1', 
               OPENBLAS_NUM_THREADS='1')
    commands = {}
    for model in models:
        (output / 'logs' / model).mkdir(parents=True, exist_ok=True)
        model_commands = {}
        for case in ['full', *MIA_COLUMNS]:
            command = [sys.executable, '-u', str(scripts / 'run_mia_ablation.py'),
                       '--model_name', model, '--mia_variant', case,
                       '--metadata_root', metadata_root, '--feature_root', feature_root,
                       '--main_results', main_results,
                       '--local_dataset_path', local_dataset_path,
                       '--dataset_total_num', str(args.dataset_total_num),
                       '--num_pairs', str(args.num_pairs), '--k_base', str(args.k_base),
                       '--split_times', str(args.split_times), '--global_seed', str(args.global_seed),
                       '--output_root', str(output / model / case)]
            commands[model, case] = command
            model_commands[case] = command
        (output / f'commands_{model}.json').write_text(json.dumps(model_commands, indent=2) + '\n')

    def run(job):
        model, case = job
        with (output / 'logs' / model / f'{case}.log').open('a') as log:
            result = subprocess.run(commands[job], env=env, stdout=log, stderr=subprocess.STDOUT)
        return job, result.returncode

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in commands]
        for future in as_completed(futures):
            job, code = future.result()
            print(f'{job[0]}/{job[1]}: exit {code}', flush=True)
            if code:
                failed.append(job)
    if failed:
        raise RuntimeError(f'Ablation cases failed: {failed}')
    subprocess.run([sys.executable, str(scripts / 'export_mia_ablation.py'),
                    '--results', str(output), '--output', str(output / 'summary_all_models.csv')], check=True)


if __name__ == '__main__':
    main()
