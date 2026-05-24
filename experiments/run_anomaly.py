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
    python experiments/run_anomaly.py --partition anne inne
    python experiments/run_anomaly.py --partition anne,inne,iforest
"""

import os, sys, time, argparse, warnings, traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.partitions import get_partition, PARTITION_NAMES
from data.datasets import DATASETS

PARTITIONS = ["anne", "inne", "inne-overlapping", "iforest", "sciforest"]
KERNELS = ["ik", "idk"]
N_ESTIMATORS = 200
MAX_SAMPLES = 16
RANDOM_STATE = 42
N_RUNS = 5
MAX_N = 10000  # subsample threshold
OUT_DIR = os.path.join(ROOT, "results", "anomaly_detection")
OUT_PATH = os.path.join(OUT_DIR, "auc_results.csv")
RESULT_COLUMNS = [
    "dataset",
    "partition",
    "partition_name",
    "kernel",
    "timestamp",
    "n",
    "features",
    "shape",
    "density",
    "dim_level",
    "size_level",
    "condition",
    "source",
    "anom_rate",
    "auc_mean",
    "auc_std",
    "auc_min",
    "auc_max",
    "fit_time_s",
    "transform_time_s",
    "score_time_s",
    "total_time_s",
    "fit_time_std",
    "auc_per_sec",
    "phi_width",
    "phi_ones_per_point_per_estimator",
    "phi_ones_per_point_per_estimator_normal",
    "phi_ones_per_point_per_estimator_anomaly",
    "n_estimators",
    "max_samples",
    "n_runs",
]


def _run_anomaly_task(task):
    """Worker: run one (dataset, partition, kernel) combination."""
    ds_name, partition, kernel, n_est, max_samples = task
    from data.datasets import DATASETS
    ds = DATASETS[ds_name]
    return run_one(ds, partition, kernel, n_est, max_samples)


def _get_auc_consistent_masks(y):
    """Return normal/anomaly masks aligned with roc_auc_score binary convention.

    For binary targets, sklearn uses the larger/sorted-second label as positive.
    """
    y = np.asarray(y)
    labels = np.unique(y)
    if labels.size != 2:
        return None, None
    positive_label = labels[1]
    anomaly_mask = y == positive_label
    normal_mask = ~anomaly_mask
    return normal_mask, anomaly_mask


def _load_existing(path, n_estimators):
    """Load existing CSV and return set of completed keys for current n_estimators."""
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_csv(path)
        # Only consider rows with matching n_estimators so fast/full can coexist
        df_match = df[df["n_estimators"] == n_estimators]
        keys = set(
            zip(
                df_match["dataset"],
                df_match["partition"],
                df_match["kernel"],
                df_match["n_estimators"],
            )
        )
        return keys, df.to_dict("records")
    except Exception:
        return set(), []


def _append_row(row, path):
    """Append a single row to CSV, creating header only if file is empty."""
    df = pd.DataFrame([row]).reindex(columns=RESULT_COLUMNS)
    header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not header:
        existing_columns = list(pd.read_csv(path, nrows=0).columns)
        if existing_columns != RESULT_COLUMNS:
            existing = pd.read_csv(path)
            for column in RESULT_COLUMNS:
                if column not in existing.columns:
                    existing[column] = np.nan
            existing = existing.reindex(columns=RESULT_COLUMNS)
            existing.to_csv(path, index=False)
    df.to_csv(path, mode="a", header=header, index=False)


def run_one(ds, partition_method, kernel, n_estimators, max_samples):
    X = ds["X"].astype(np.float32)
    y = ds["y"]

    if len(X) > MAX_N:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[idx], y[idx]

    aucs = []
    fit_times, transform_times, score_times = [], [], []
    phi_widths = []
    phi_ones_per_point_per_estimator = []
    phi_ones_per_point_per_estimator_normal = []
    phi_ones_per_point_per_estimator_anomaly = []

    anomaly_mask = y == 1
    normal_mask = y == 0

    for run in range(N_RUNS):
        part = get_partition(
            partition_method,
            kernel=kernel,
            n_estimators=n_estimators,
            max_samples=max_samples,
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
        phi_ones_per_point_per_estimator.append(
            float(phi.nnz) / float(phi.shape[0] * n_estimators)
        )
        ones_per_point_per_estimator = np.diff(phi.indptr).astype(np.float64) / float(
            n_estimators
        )
        if np.any(normal_mask):
            phi_ones_per_point_per_estimator_normal.append(
                float(np.mean(ones_per_point_per_estimator[normal_mask]))
            )
        else:
            phi_ones_per_point_per_estimator_normal.append(float("nan"))

        if np.any(anomaly_mask):
            phi_ones_per_point_per_estimator_anomaly.append(
                float(np.mean(ones_per_point_per_estimator[anomaly_mask]))
            )
        else:
            phi_ones_per_point_per_estimator_anomaly.append(float("nan"))

        # ── anomaly scores ────────────────────────────────────────
        t0 = time.perf_counter()
        if kernel == "ik":
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
            auc = float("nan")

        aucs.append(auc)
        fit_times.append(fit_t)
        transform_times.append(transform_t)
        score_times.append(score_t)

    return {
        # ── identity ──────────────────────────────────────────────
        "dataset": ds["name"],
        "partition": partition_method,
        "partition_name": PARTITION_NAMES[partition_method],
        "kernel": kernel,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # ── dataset properties ────────────────────────────────────
        "n": len(X),
        "features": ds["features"],
        "shape": ds["shape"],
        "density": ds["density"],
        "dim_level": ds["dim_level"],
        "size_level": ds["size_level"],
        "condition": ds["condition"],
        "source": ds["source"],
        "anom_rate": round(float(np.mean(y == 1)) * 100.0, 2),
        # ── performance metrics ───────────────────────────────────
        "auc_mean": round(float(np.nanmean(aucs)), 4),
        "auc_std": round(float(np.nanstd(aucs)), 4),
        "auc_min": round(float(np.nanmin(aucs)), 4),
        "auc_max": round(float(np.nanmax(aucs)), 4),
        # ── timing breakdown (mean over runs) ────────────────────
        "fit_time_s": round(float(np.mean(fit_times)), 4),
        "transform_time_s": round(float(np.mean(transform_times)), 4),
        "score_time_s": round(float(np.mean(score_times)), 4),
        "total_time_s": round(
            float(
                np.mean(
                    [
                        f + t + s
                        for f, t, s in zip(fit_times, transform_times, score_times)
                    ]
                )
            ),
            4,
        ),
        "fit_time_std": round(float(np.std(fit_times)), 4),
        # ── efficiency metrics ────────────────────────────────────
        "auc_per_sec": round(
            float(np.nanmean(aucs))
            / (
                float(
                    np.mean(
                        [
                            f + t + s
                            for f, t, s in zip(fit_times, transform_times, score_times)
                        ]
                    )
                )
                + 1e-9
            ),
            4,
        ),
        # ── partition characteristics ─────────────────────────────
        "phi_width": int(np.mean(phi_widths)),
        "phi_ones_per_point_per_estimator": round(
            float(np.mean(phi_ones_per_point_per_estimator)), 4
        ),
        "phi_ones_per_point_per_estimator_normal": round(
            float(np.nanmean(phi_ones_per_point_per_estimator_normal)), 4
        ),
        "phi_ones_per_point_per_estimator_anomaly": round(
            float(np.nanmean(phi_ones_per_point_per_estimator_anomaly)), 4
        ),
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "n_runs": N_RUNS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--partition",
        nargs="+",
        default=None,
        help="One or more partitions (space-separated and/or comma-separated)",
    )
    args = parser.parse_args()

    n_est = 50 if args.fast else N_ESTIMATORS
    os.makedirs(OUT_DIR, exist_ok=True)

    ad_datasets = []
    for ds in DATASETS.values():
        if ds["task"] != "AD" or (args.dataset is not None and ds["name"] != args.dataset):
            continue
        try:
            _ = ds["X"]  # trigger lazy load once
            ad_datasets.append(ds)
        except Exception as e:
            print(f"  SKIP {ds['name']:28s} (load error: {e})")

    if args.partition:
        tokens = [
            p.strip() for arg in args.partition for p in arg.split(",") if p.strip()
        ]
        invalid = sorted(set(tokens) - set(PARTITIONS))
        if invalid:
            parser.error(
                f"Unknown partition(s): {', '.join(invalid)}. "
                f"Valid options: {', '.join(PARTITIONS)}"
            )
        partitions = list(dict.fromkeys(tokens))
    else:
        partitions = PARTITIONS

    print("=" * 68)
    print("  IK Partitioning Study — Anomaly Detection Experiments")
    print("=" * 68)
    print(f"  Datasets   : {len(ad_datasets)}")
    print(f"  Partitions : {partitions}")
    print(f"  Kernels    : {KERNELS}")
    print(f"  n_est      : {n_est}   psi : {MAX_SAMPLES}")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  Output     : {OUT_DIR}")
    print("=" * 68)
    print()

    completed, results = _load_existing(OUT_PATH, n_est)
    total = len(ad_datasets) * len(partitions) * len(KERNELS)
    done = len(results)
    skipped = 0

    tasks = []
    for ds in ad_datasets:
        for partition in partitions:
            for kernel in KERNELS:
                key = (ds["name"], partition, kernel, n_est)
                if key in completed:
                    skipped += 1
                    continue
                tasks.append((ds["name"], partition, kernel, n_est, MAX_SAMPLES))

    total = len(tasks) + skipped
    done = skipped
    workers = max(1, min(8, len(tasks)))  # cap threads, but always at least 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_anomaly_task, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            row = future.result()
            if row is None:
                continue
            results.append(row)
            _append_row(row, OUT_PATH)
            print(
                f"  [{done:3d}/{total}] {row['dataset']:28s} {row['partition']:12s} {row['kernel']}  "
                f"AUC={row['auc_mean']:.3f}+/-{row['auc_std']:.3f}  "
                f"phi={row['phi_width']}  "
                f"ones/pt/est={row['phi_ones_per_point_per_estimator']:.2f} "
                f"(n={row['phi_ones_per_point_per_estimator_normal']:.2f}, "
                f"a={row['phi_ones_per_point_per_estimator_anomaly']:.2f})  "
                f"t={row['total_time_s']:.1f}s"
            )

    if skipped:
        print(f"\n  Skipped {skipped} already-completed combinations.")

    df = pd.DataFrame(results)
    print(f"\n  Saved {len(df)} rows -> {OUT_PATH}")

    if len(df) == 0:
        return

    print("\n  Mean AUC per partition x kernel:")
    print(
        df.groupby(["partition", "kernel"])["auc_mean"]
        .mean()
        .unstack()
        .round(3)
        .to_string()
    )
    print("\n  Mean total_time_s per partition:")
    print(df.groupby("partition")["total_time_s"].mean().round(3).to_string())
    print("\n  Mean AUC per condition x partition:")
    print(
        df.groupby(["condition", "partition"])["auc_mean"]
        .mean()
        .unstack()
        .round(3)
        .to_string()
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
