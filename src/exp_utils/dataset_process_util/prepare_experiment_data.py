"""
Select source images and save their metadata for the complete experiment.
"""
import argparse
import json
from pathlib import Path
import random


def prepare_sources(dataset_root, output_root, seed):
    metadata_root = output_root / 'experiments/shared_res/datasets'
    sources = {}
    for name in ['laion_aesthetics_v2_5p0', 'commoncatalog_cc_by_sa',
                 'coco_2017_val', 'nocaps', 'laion2b_multi_translated']:
        split = 'train' if name in ['laion_aesthetics_v2_5p0', 'commoncatalog_cc_by_sa'] else 'val'
        count = 1000 if split == 'train' else 3000
        source = dataset_root / name
        files = sorted((source / 'data' / split).glob('*.jpg'), key=lambda p: int(p.stem))
        indices = [int(p.stem) for p in random.Random(seed).sample(files, count)]
        with (source / 'metadata' / f'meta_{name}.jsonl').open() as f:
            captions = {int(json.loads(line)['index']) for line in f if line.strip()}
        missing = set(indices) - captions
        if missing:
            raise ValueError(f'Missing captions for {name}: {sorted(missing)}')
        sources[name] = indices
        target = metadata_root / name / 'metadata' / f'meta_{name}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('w') as f:
            json.dump([{'source_dataset': name, 'original_index': i} for i in indices], f, indent=4)
    with (metadata_root / 'sample_manifest.json').open('w') as f:
        json.dump({'source_indices': sources}, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local_dataset_path', type=Path, required=True)
    parser.add_argument('--output_root', type=Path, default=Path(__file__).resolve().parents[3] / 'exp_data')
    parser.add_argument('--global_seed', type=int, default=12345)
    args = parser.parse_args()
    prepare_sources(args.local_dataset_path, args.output_root, args.global_seed)
