"""
Draw Figure 7 from measured RQ6 rejection rates.
"""
import pandas as pd
import matplotlib.pyplot as plt

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
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = 36
plt.rcParams['axes.labelsize'] = 32
plt.rcParams['axes.titlesize'] = 26
plt.rcParams['legend.fontsize'] = 26
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26

def plot_rq6(csv_path, output_path):

    data = pd.read_csv(csv_path)

    data['DUL'] = abs(data['FPR'] - 0.001) + (1 - data['TPR'])

    plt.figure(figsize=(10, 6))

    plt.plot(data['Dataset_Size'], data['TPR'], marker='o', label='TPR', linewidth=2, markersize=6)
    plt.plot(data['Dataset_Size'], data['FPR'], marker='s', label='FPR', linewidth=2, markersize=6)
    plt.plot(data['Dataset_Size'], data['DUL'], marker='^', label='DUL', linewidth=2, markersize=6)

    plt.xlabel('Single Dataset Size')
    plt.ylabel('Values')

    plt.legend(loc='best', labels=['TPR', 'FPR', 'DUL'])

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Plot RQ6 detection rates across dataset sizes")
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New PDF file')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_rq6(args.csv, args.output)
