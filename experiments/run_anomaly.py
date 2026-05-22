"""
experiments/run_anomaly.py
===========================
Anomaly detection experiments: 4 partitions x IK + IDK x all AD datasets.

Saves to: results/anomaly_detection/auc_results.csv

Usage:
    python experiments/run_anomaly.py
    python experiments/run_anomaly.py --fast
    python experiments/run_anomaly.py --dataset thyroid
    python experiments/run_anomaly.py --partition anne
"""

import os, sys, time, argparse, warnings, traceback
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.partitions import get_partition, PARTITION_NAMES
from data.datasets   import DATASETS

PARTITIONS   = ['anne', 'inne', 'iforest', 'sciforest']
KERNELS      = ['ik', 'idk']
N_ESTIMATORS = 200
MAX_SAMPLES  = 16
RANDOM_STATE = 42
N_RUNS       = 5
MAX_N        = 10000      # subsample threshold
OUT_DIR      = os.path.join(ROOT, 'results', 'anomaly_detection')
OUT_PATH     = os.path.join(OUT_DIR, 'auc_results.csv')


def _load_existing(path, n_estimators):
    """Load existing CSV and return set of completed keys for current n_estimators."""
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_csv(path)
        # Only consider rows with matching n_estimators so fast/full can coexist
        df_match = df[df['n_estimators'] == n_estimators]
        keys = set(zip(df_match['dataset'], df_match['partition'],
                       df_match['kernel'], df_match['n_estimators']))
        return keys, df.to_dict('records')
    except Exception:
        return set(), []


def _append_row(row, path):
    """Append a single row to CSV, creating header only if file is empty."""
    df = pd.DataFrame([row])
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    df.to_csv(path, mode='a', header=header, index=False)


def run_one(ds, partition_method, kernel, n_estimators, max_samples):
    X = ds['X'].astype(np.float32)
    y = ds['y']

    if len(X) > MAX_N:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[idx], y[idx]

    aucs = []
    fit_times, transform_times, score_times = [], [], []
    phi_widths = []

    for run in range(N_RUNS):
        part = get_partition(
            partition_method, kernel=kernel,
            n_estimators=n_estimators, max_samples=max_samples,
            random_state=RANDOM_STATE + run,
        )

        # ── fit ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        part.fit(X)
        fit_t = time.perf_counter() - t0

        # ── transform ────────────────────────────────────────────
        t0 = time.perf_counter()
        phi = part.transform(X)
        transform_t = time.perf_counter() - t0
        phi_widths.append(phi.shape[1])

        # ── anomaly scores ────────────────────────────────────────
        t0 = time.perf_counter()
        if kernel == 'ik':
            # IK score: 1 - mean similarity to all other points
            K = part.similarity_ik(X)
            scores = 1.0 - K.mean(axis=1)
        else:
            # IDK score: 1 - similarity to global distribution
            scores = part.idk_scores(X)
        score_t = time.perf_counter() - t0

        try:
            auc = roc_auc_score(y, scores)
        except Exception:
            auc = float('nan')

        aucs.append(auc)
        fit_times.append(fit_t)
        transform_times.append(transform_t)
        score_times.append(score_t)

    return {
        # ── identity ──────────────────────────────────────────────
        'dataset':          ds['name'],
        'partition':        partition_method,
        'partition_name':   PARTITION_NAMES[partition_method],
        'kernel':           kernel,
        'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        # ── dataset properties ────────────────────────────────────
        'n':                len(X),
        'features':         ds['features'],
        'shape':            ds['shape'],
        'density':          ds['density'],
        'dim_level':        ds['dim_level'],
        'size_level':       ds['size_level'],
        'condition':        ds['condition'],
        'source':           ds['source'],
        'anom_rate':        ds.get('anom_rate') if 'AD' in ds['task'] else None,
        # ── performance metrics ───────────────────────────────────
        'auc_mean':         round(float(np.nanmean(aucs)), 4),
        'auc_std':          round(float(np.nanstd(aucs)),  4),
        'auc_min':          round(float(np.nanmin(aucs)),  4),
        'auc_max':          round(float(np.nanmax(aucs)),  4),
        # ── timing breakdown (mean over runs) ────────────────────
        'fit_time_s':       round(float(np.mean(fit_times)),       4),
        'transform_time_s': round(float(np.mean(transform_times)), 4),
        'score_time_s':     round(float(np.mean(score_times)),     4),
        'total_time_s':     round(float(np.mean(
                                [f+t+s for f,t,s in
                                 zip(fit_times,transform_times,score_times)])), 4),
        'fit_time_std':     round(float(np.std(fit_times)),        4),
        # ── efficiency metrics ────────────────────────────────────
        'auc_per_sec':      round(float(np.nanmean(aucs)) / (float(np.mean(
                                [f+t+s for f,t,s in
                                 zip(fit_times,transform_times,score_times)])) + 1e-9), 4),
        # ── partition characteristics ─────────────────────────────
        'phi_width':        int(np.mean(phi_widths)),
        'n_estimators':     n_estimators,
        'max_samples':      max_samples,
        'n_runs':           N_RUNS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast',      action='store_true')
    parser.add_argument('--dataset',   type=str, default=None)
    parser.add_argument('--partition', type=str, default=None)
    args = parser.parse_args()

    n_est = 50 if args.fast else N_ESTIMATORS
    os.makedirs(OUT_DIR, exist_ok=True)

    ad_datasets = [ds for ds in DATASETS.values()
                   if ds['task'] == 'AD'
                   and (args.dataset is None or ds['name'] == args.dataset)]
    partitions  = [args.partition] if args.partition else PARTITIONS

    print('=' * 68)
    print('  IK Partitioning Study — Anomaly Detection Experiments')
    print('=' * 68)
    print(f'  Datasets   : {len(ad_datasets)}')
    print(f'  Partitions : {partitions}')
    print(f'  Kernels    : {KERNELS}')
    print(f'  n_est      : {n_est}   ψ : {MAX_SAMPLES}')
    print(f'  Runs each  : {N_RUNS}')
    print(f'  Output     : {OUT_DIR}')
    print('=' * 68)
    print()

    completed, results = _load_existing(OUT_PATH, n_est)
    total   = len(ad_datasets) * len(partitions) * len(KERNELS)
    done    = len(results)
    skipped = 0

    for ds in ad_datasets:
        for partition in partitions:
            for kernel in KERNELS:
                key = (ds['name'], partition, kernel, n_est)
                if key in completed:
                    skipped += 1
                    continue
                done += 1
                tag = (f'[{done:3d}/{total}] {ds["name"]:28s} '
                       f'{partition:12s} {kernel}')
                print(f'  {tag}', end='  ', flush=True)
                try:
                    t0  = time.perf_counter()
                    row = run_one(ds, partition, kernel, n_est, MAX_SAMPLES)
                    elapsed = time.perf_counter() - t0
                    results.append(row)
                    _append_row(row, OUT_PATH)
                    print(f'AUC={row["auc_mean"]:.3f}±{row["auc_std"]:.3f}  '
                          f'φ={row["phi_width"]}  '
                          f't={elapsed:.1f}s')
                except Exception as e:
                    print(f'FAILED — {e}')
                    traceback.print_exc()

    if skipped:
        print(f'\n  Skipped {skipped} already-completed combinations.')

    df = pd.DataFrame(results)
    print(f'\n  Saved {len(df)} rows → {OUT_PATH}')

    if len(df) == 0:
        return

    print('\n  Mean AUC per partition x kernel:')
    print(df.groupby(['partition','kernel'])['auc_mean'].mean().unstack().round(3).to_string())
    print('\n  Mean total_time_s per partition:')
    print(df.groupby('partition')['total_time_s'].mean().round(3).to_string())
    print('\n  Mean AUC per condition x partition:')
    print(df.groupby(['condition','partition'])['auc_mean'].mean().unstack().round(3).to_string())
    print('\nDone.')


if __name__ == '__main__':
    main()