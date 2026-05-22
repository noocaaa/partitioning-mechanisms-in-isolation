"""
notebooks/visualize_partitions.py
=================================
Generate partition geometry & kernel visualizations for all 2D datasets.

Usage:
    python notebooks/visualize_partitions.py
    python notebooks/visualize_partitions.py --dataset syn_moons_small
    python notebooks/visualize_partitions.py --fast

Saves figures to: results/figures/
"""

import os, sys, argparse, warnings

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from data.datasets import DATASETS
from src.partitions import get_partition
from src.visualize import plot_all_partitions, plot_full_report, plot_phi_spy
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(ROOT, 'results', 'figures')
N_EST   = 200
MAX_SAMPLES = 16
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--n_estimators', type=int, default=None)
    args = parser.parse_args()

    n_est = args.n_estimators or (50 if args.fast else N_EST)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Select 2D datasets only
    candidates = [ds for ds in DATASETS.values()
                  if ds['features'] == 2
                  and (args.dataset is None or ds['name'] == args.dataset)]

    if not candidates:
        print('No 2D datasets found.')
        return

    print('=' * 60)
    print('  Partition Visualisation')
    print(f'  Datasets : {[d["name"] for d in candidates]}')
    print(f'  n_est    : {n_est}')
    print(f'  Output   : {OUT_DIR}')
    print('=' * 60)

    for ds in candidates:
        name = ds['name']
        X = ds['X'].astype(np.float32)
        y = ds['y']
        print(f'\n  {name}  n={len(X)}')

        partitions = {}
        kernels = {}
        scores = {}

        for method in ['anne', 'inne', 'iforest', 'sciforest']:
            print(f'    {method:12s} ...', end=' ', flush=True)
            part = get_partition(method, n_estimators=n_est,
                                 max_samples=MAX_SAMPLES,
                                 random_state=RANDOM_STATE)
            part.fit(X)
            partitions[method] = part
            kernels[method] = {
                'ik':  part.similarity_ik(X),
                'idk': part.similarity_idk(X),
            }
            scores[method] = part.idk_scores(X)
            print('OK')

        # 1. Geometry comparison
        print('    Saving geometry comparison ...', end=' ')
        plot_all_partitions(
            X, y, partitions, dataset_name=name,
            save_path=os.path.join(OUT_DIR, f'{name}_geometries.png')
        )
        plt.close('all')
        print('OK')

        # 2. Full report (geometry + kernels + scores)
        print('    Saving full report ...', end=' ')
        plot_full_report(
            X, y, partitions, dataset_name=name,
            kernels=kernels, scores=scores,
            save_path=os.path.join(OUT_DIR, f'{name}_report.png')
        )
        plt.close('all')
        print('OK')

        # 3. Phi maps
        print('    Saving phi maps ...', end=' ')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, (method, part) in zip(axes.ravel(), partitions.items()):
            phi = part.transform(X)
            plot_phi_spy(ax, phi, title=f'Φ — {method}')
        fig.suptitle(f'{name} — Feature maps', fontsize=12)
        plt.tight_layout()
        phi_path = os.path.join(OUT_DIR, f'{name}_phi_maps.png')
        fig.savefig(phi_path, dpi=200, bbox_inches='tight')
        plt.close('all')
        print('OK')

    print(f'\n  Done. Figures saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
