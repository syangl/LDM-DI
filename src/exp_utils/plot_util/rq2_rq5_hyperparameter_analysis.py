"""
Compute RQ2/RQ5 calibration grids and draw their contour plots.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

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
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

TITLE_FONT_SIZE = 36
AXES_LABEL_FONT_SIZE = 38
TICK_FONT_SIZE = 34
CBAR_FONT_SIZE = 32
CONTOUR_LABEL_FONT_SIZE = 34

K_COLORS = ['#F58231', '#FFE119', '#4363D8', '#E6194B']
K_LEVELS = [50, 100, 200, 400,]
K_LINESTYLES = ['dashed', 'dashdot', 'dotted', 'solid']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calc_semantic_shift(emb_a, emb_b):
    X = np.vstack([emb_a, emb_b])
    y = np.hstack([np.ones(len(emb_a)), np.zeros(len(emb_b))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, shuffle=True, stratify=y)
    
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE).unsqueeze(1)
    X_te = torch.FloatTensor(X_test).to(DEVICE)
    y_te = torch.FloatTensor(y_test).to(DEVICE).unsqueeze(1)
    
    model = nn.Sequential(nn.Linear(X.shape[1], 1), nn.Sigmoid()).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    model.train()
    for _ in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        acc = ((model(X_te) > 0.5).float() == y_te).float().mean().item()
    return 2 * abs(2 * acc - 1)

def calc_texture_shift(lap_a, lap_b, fft_a, fft_b):
    shifts = []
    for a, b in [(lap_a, lap_b), (fft_a, fft_b)]:
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 50)
        hist1, _ = np.histogram(a, bins=bins, density=True)
        hist2, _ = np.histogram(b, bins=bins, density=True)
        shifts.append(1.0 - np.sum(np.minimum(hist1, hist2)) * (bins[1] - bins[0]))
    return max(shifts)

def update_texture_shift(pool_df, prepared_df, base_dataset_path):
    prepared_df = prepared_df.copy()
    prepared_df['original_index'] = prepared_df['original_index'].astype(int)
    features = prepared_df.drop_duplicates(['source_dataset', 'original_index']).set_index(['source_dataset', 'original_index'])
    shifts = []
    for pair_id in tqdm(pool_df['pair_id'], desc='Recomputing textural shifts'):
        groups = []
        for side in [1, 2]:
            with (base_dataset_path / f'pair_{pair_id}_base{side}.json').open() as f:
                metadata = json.load(f)
            keys = [(item['source_dataset'], int(item['original_index'])) for item in metadata]
            groups.append(features.loc[keys])
        shifts.append(calc_texture_shift(groups[0]['laplacian'].to_numpy(), groups[1]['laplacian'].to_numpy(),
                                         groups[0]['fft'].to_numpy(), groups[1]['fft'].to_numpy()))
    pool_df = pool_df.copy()
    pool_df['delta_txt'] = shifts
    return pool_df

def process_real_pool(prepared_df, base_dataset_path, ttest_dir, case):
    records = []
    prepared_df = prepared_df.copy()
    prepared_df['original_index'] = prepared_df['original_index'].astype(int)
    features = prepared_df.drop_duplicates(['source_dataset', 'original_index']).set_index(['source_dataset', 'original_index'])
    pair_files = [f for f in os.listdir(base_dataset_path) if f.endswith("_base1.json")]
    pair_files = [f for f in pair_files if (1 <= int(f.split('_')[1]) <= 500 if case == 'rq2'
                                          else 501 <= int(f.split('_')[1]) <= 1500)]
    
    print("[*] Processing Real Deltas & T-Stats...")
    for f in tqdm(pair_files):
        pair_idx = int(f.split('_')[1])
        ttest_file = os.path.join(ttest_dir, f"pair_{pair_idx}", f"ttest_pair_{pair_idx}_results.json")
        if not os.path.exists(ttest_file):
            raise FileNotFoundError(ttest_file)
            
        with open(ttest_file, 'r') as tf:
            res = json.load(tf)
            wt_null = res[0]['t_statistic'] if isinstance(res, list) else res['t_statistic']
            
        with open(os.path.join(base_dataset_path, f"pair_{pair_idx}_base1.json"), 'r') as m1:
            meta1 = json.load(m1)
        with open(os.path.join(base_dataset_path, f"pair_{pair_idx}_base2.json"), 'r') as m2:
            meta2 = json.load(m2)
            
        emb1, lap1, fft1 = [], [], []
        for item in meta1:
            match = features.loc[(item['source_dataset'], int(item['original_index']))]
            emb1.append(match['embedding'])
            lap1.append(match['laplacian'])
            fft1.append(match['fft'])
            
        emb2, lap2, fft2 = [], [], []
        for item in meta2:
            match = features.loc[(item['source_dataset'], int(item['original_index']))]
            emb2.append(match['embedding'])
            lap2.append(match['laplacian'])
            fft2.append(match['fft'])
            
        delta_sem = calc_semantic_shift(np.array(emb1), np.array(emb2))
        delta_txt = calc_texture_shift(np.array(lap1), np.array(lap2), np.array(fft1), np.array(fft2))
        
        records.append({'pair_id': pair_idx, 'delta_sem': delta_sem, 'delta_txt': delta_txt, 'wt_null': wt_null})
        
    return pd.DataFrame(records)

def plot_clean_adaptive_basin(grid_df, ax, fig, title="", is_first_col=False, is_last_col=False, alpha=0.001):
    pivot_loss = grid_df.pivot(index='u_txt', columns='u_sem', values='du_loss')
    pivot_k = grid_df.pivot(index='u_txt', columns='u_sem', values='k_base')
    
    X = pivot_loss.columns.values
    Y = pivot_loss.index.values
    Z_loss = pivot_loss.values
    Z_k = pivot_k.values
    
    cmap = plt.cm.Greens_r

    fixed_levels = np.linspace(0.0, 1.0, 51)
    ax.set_facecolor('white')
    c = ax.contourf(X, Y, np.ma.masked_invalid(Z_loss), levels=fixed_levels, cmap=cmap, extend='max')

    cs_k = ax.contour(X, Y, Z_k, levels=K_LEVELS, colors=K_COLORS, linewidths=4, linestyles=K_LINESTYLES, alpha=0.9)

    constrained_df = grid_df[grid_df['k_base'] <= 500]
    
    if is_last_col:
        cbar = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Detection Utility Loss', fontsize=CBAR_FONT_SIZE, labelpad=15)
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax.set_xlabel(r'Semantic $U_{\mathrm{sem}}$', fontsize=AXES_LABEL_FONT_SIZE, labelpad=15)
    if is_first_col:
        ax.set_ylabel(r'Textural $U_{\mathrm{txt}}$', fontsize=AXES_LABEL_FONT_SIZE, labelpad=15)
        
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE, 
                   top=True, right=True, direction='in')

    ax.text(0.5, -0.30, title, transform=ax.transAxes, 
            ha='center', va='top', fontsize=TITLE_FONT_SIZE, fontweight='bold')

def compute_grid(pool_df, target_wts, member_wts, u_sem_grid, u_txt_grid, alpha=0.001):
    """
    Apply percentile calibration at each pair of shift bounds.
    """
    results = []
    for u_sem in u_sem_grid:
        for u_txt in u_txt_grid:
            valid_pairs = pool_df[(pool_df['delta_sem'] <= u_sem) & (pool_df['delta_txt'] <= u_txt)]
            k_base = len(valid_pairs)
            if k_base == 0:
                results.append((u_sem, u_txt, 0, np.nan, np.nan, np.nan))
                continue
            tau_alpha = np.percentile(valid_pairs['wt_null'].values, 100 * (1 - alpha))
            fpr = np.sum(target_wts > tau_alpha) / len(target_wts)
            tpr = np.sum(member_wts > tau_alpha) / len(member_wts)
            du_loss = abs(fpr - alpha) + (1 - tpr)
            results.append((u_sem, u_txt, k_base, fpr, tpr, du_loss))
    return pd.DataFrame(results, columns=['u_sem', 'u_txt', 'k_base', 'fpr', 'tpr', 'du_loss'])

def plot_grid(results_root, output_root, case, verify_saved=False):
    """
    Build or reuse measured pools, compute a grid, and draw its contour figure.
    """
    from pathlib import Path
    root = Path(results_root)
    section = 'RQ2_hyperparameters' if case == 'rq2' else 'RQ5_distribution_bound_ablation'
    shared = root / 'experiments/shared_res/RQ2_RQ5_calibration_pool'
    output_root = Path(output_root)
    output_dir = output_root / 'plot_analyze' / section
    output_dir.mkdir(parents=True, exist_ok=False)
    pool_output_dir = output_root / 'experiments' / section
    models = ['stable-diffusion-v1p5', 'commoncanvas-s-c', 'kandinsky-3']
    titles = ['Stable-Diffusion-v1-5', 'CommonCanvas-S-C', 'Kandinsky-3']
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=False)
    for i, (model, title) in enumerate(zip(models, titles)):
        pool_path = root / 'experiments' / section / f'real_pool_{model}.csv'
        output_pool = pool_output_dir / pool_path.name
        ttest_dir = shared / 'intermediate' / model / 'ttest'
        if pool_path.exists():
            pool_df = pd.read_csv(pool_path)
        else:
            if verify_saved:
                raise FileNotFoundError(pool_path)
            prepared = pd.read_pickle(shared / 'intermediate/prepared_df.pkl')
            pool_df = process_real_pool(prepared, shared / 'datasets/D_base', ttest_dir, case)
        if case == 'rq2':
            pool_df = pool_df[pool_df['pair_id'].between(1, 500)]
        else:
            pool_df = pool_df[pool_df['pair_id'].between(501, 1500)]
        pool_df = pool_df.sample(n=500, random_state=0)
        if not verify_saved and pool_path.exists():
            prepared = pd.read_pickle(shared / 'intermediate/prepared_df.pkl')
            pool_df = update_texture_shift(pool_df, prepared, shared / 'datasets/D_base')
        pool_output_dir.mkdir(parents=True, exist_ok=True)
        pool_df.to_csv(output_pool, index=False, mode='x')
        with (ttest_dir / 'pair_holdhold/ttest_pair_holdhold_results.json').open() as f:
            target_wts = np.array([r['t_statistic'] for r in json.load(f)])
        with (ttest_dir / 'pair_sushold/ttest_pair_sushold_results.json').open() as f:
            member_wts = np.array([r['t_statistic'] for r in json.load(f)])
        if verify_saved:
            saved = pd.read_csv(root / 'plot_analyze' / section / f'grid_search_{model}.csv')
            # Use stored grid coordinates to retain exact <= boundary comparisons.
            u_sem_grid = sorted(saved.u_sem.unique())
            u_txt_grid = sorted(saved.u_txt.unique())
        else:
            u_sem_grid = np.linspace(pool_df.delta_sem.min(), pool_df.delta_sem.max(), 30)
            u_txt_grid = np.linspace(0.04, 0.15, 30) if case == 'rq2' else np.linspace(pool_df.delta_txt.min(), pool_df.delta_txt.max(), 30)
        grid = compute_grid(pool_df, target_wts, member_wts, u_sem_grid, u_txt_grid)
        if verify_saved:
            coordinates_match = np.allclose(grid[['u_sem', 'u_txt', 'k_base']], saved[['u_sem', 'u_txt', 'k_base']], rtol=1e-12, atol=1e-12)
            valid = grid['k_base'] > 0
            metrics_match = np.allclose(grid.loc[valid, ['fpr', 'tpr', 'du_loss']], saved.loc[valid, ['fpr', 'tpr', 'du_loss']], rtol=1e-12, atol=1e-12)
            if not coordinates_match or not metrics_match:
                raise ValueError(f'Recomputed grid differs from saved {case}: {model}')
        grid.to_csv(output_dir / f'grid_search_{model}.csv', index=False)
        plot_clean_adaptive_basin(grid, axes[i], fig, f'({chr(97+i)}) {title}', i == 0, i == 2)
        if case == 'rq2':
            axes[i].set_ylim(0.10, 0.15)
            axes[i].tick_params(axis='x', pad=12)
    handles = [mlines.Line2D([], [], linestyle='none', label=r'$K_{\mathrm{base}}$')]
    handles += [mlines.Line2D([], [], color=c, linewidth=4, linestyle=ls, label=f'${k}$')
                for c, k, ls in zip(K_COLORS, K_LEVELS, K_LINESTYLES)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.89),
               ncol=len(handles), fontsize=CONTOUR_LABEL_FONT_SIZE,
               frameon=False, handletextpad=0.2)
    fig.subplots_adjust(top=0.88, bottom=0.24, left=0.08, right=0.92, wspace=0.25)
    filename = 'fig_heatmap_3models_combined.pdf' if case == 'rq2' else 'fig_rq5_heatmap_3models_combined.pdf'
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / filename.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Generate RQ2/RQ5 pools and grids, or verify saved measurements')
    parser.add_argument('--results', type=Path, required=True, help='Input root containing experiments and plot_analyze')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[3] / 'exp_data', help='Output root containing experiments and plot_analyze')
    parser.add_argument('--case', choices=['rq2', 'rq5', 'both'], default='both')
    parser.add_argument('--verify-saved', action='store_true', help='Compare computed values with the input grid coordinates and measurements')
    args = parser.parse_args()
    for case in (['rq2', 'rq5'] if args.case == 'both' else [args.case]):
        plot_grid(args.results, args.output, case, args.verify_saved)
