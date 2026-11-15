import re

def parse_time(time_str):
    """
    Parse a duration containing hours, minutes and seconds.
    """
    time_str = time_str.lower()
    total_seconds = 0.0

    hours = re.search(r'(\d+(?:\.\d+)?)\s*(?:h|hour|hours)', time_str)
    minutes = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)', time_str)
    seconds = re.search(r'(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)', time_str)

    if hours:
        total_seconds += float(hours.group(1)) * 3600
    if minutes:
        total_seconds += float(minutes.group(1)) * 60
    if seconds:
        total_seconds += float(seconds.group(1))

    if total_seconds == 0.0 and not (hours or minutes or seconds):
        raise ValueError(f"Cannot parse duration: '{time_str}'")

    return total_seconds

def format_time(total_seconds):
    """
    Format seconds as hours, minutes and seconds.
    """
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = total_seconds % 60
    
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0:
        parts.append(f"{m}m")

    parts.append(f"{s:.3f}s")
    
    return " ".join(parts)

def add_times(t1_str, t2_str):
    """
    Add two formatted durations.
    """
    t1_sec = parse_time(t1_str)
    t2_sec = parse_time(t2_str)
    total_sec = t1_sec + t2_sec
    return format_time(total_sec)

def calculate_overhead(baseline_str, ours_str):
    """
    Calculate the percentage increase over the baseline duration.
    """
    baseline_sec = parse_time(baseline_str)
    ours_sec = parse_time(ours_str)
    
    if baseline_sec == 0:
        return "Baseline is 0, cannot calculate overhead."
    
    overhead_ratio = ((ours_sec - baseline_sec) / baseline_sec) * 100
    return f"+{overhead_ratio:.2f}%"

if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Summarize actual RQ4 run times')
    parser.add_argument('--offline', type=Path, required=True, help='Full feature-extraction timing JSON')
    parser.add_argument('--standard', type=Path, required=True, help='Baseline timing JSON')
    parser.add_argument('--target', type=Path, required=True, help='Calibrated target timing JSON')
    parser.add_argument('--calibration', type=Path, required=True, help='Empirical null timing JSON')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[3] / 'exp_data/plot_analyze/RQ4_time_cost/timing_summary.json', help='New timing summary JSON')
    args = parser.parse_args()
    records = {}
    for name in ['offline', 'standard', 'target', 'calibration']:
        with getattr(args, name).open() as f:
            records[name] = json.load(f)
    if records['offline']['use_10_percent_sampling'] or records['offline']['cached_datasets']:
        raise ValueError('Full-pool timing requires a full feature-extraction run')
    if any(records[name]['cached_pairs'] for name in ['target', 'calibration']):
        raise ValueError('Cached replay time cannot be reported as experimental run time')
    offline = records['offline']['total_time_seconds']
    standard = records['standard']['mean_test_seconds']
    target = records['target']['mean_test_seconds']
    calibration = records['calibration']['total_time_seconds']
    baseline_total = offline + standard
    ours_total = offline + target + calibration
    summary = {'offline_seconds': offline, 'standard_test_seconds': standard,
               'calibrated_test_seconds': target, 'calibration_seconds': calibration,
               'standard_total_seconds': baseline_total, 'ours_total_seconds': ours_total,
               'overhead_percent': (ours_total / baseline_total - 1) * 100,
               'sources': {name: str(getattr(args, name)) for name in records}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
