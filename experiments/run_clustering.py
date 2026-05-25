"""
experiments/run_clustering.py
==============================
Clustering experiments: 5 partitions x IK + IDK x all C datasets.

Saves to: results/clustering/ari_results.csv

Usage:
    python experiments/run_clustering.py
    python experiments/run_clustering.py --fast
    python experiments/run_clustering.py --t 200 --psi 256
    python experiments/run_clustering.py --dataset iris
    python experiments/run_clustering.py --partition anne
    python experiments/run_clustering.py --partition anne inne
    python experiments/run_clustering.py --partition anne,inne,iforest
"""

import os, sys, time, argparse, warnings, traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.partitions import get_partition, PARTITION_NAMES
from data.datasets import DATASETS

PARTITIONS = ["anne", "inne", "inne-overlapping", "iforest", "sciforest"]
KERNELS = ["ik", "idk"]
N_ESTIMATORS = 200
DEFAULT_MAX_SAMPLES_BY_PARTITION = {
    "anne": 16,
    "inne": 16,
    "inne-overlapping": 16,
    "iforest": 256,
    "sciforest": 256,
}
RANDOM_STATE = 42
N_RUNS = 5
MAX_N = 5000  # subsample threshold for large datasets
OUT_DIR = os.path.join(ROOT, "results", "clustering")
OUT_PATH = os.path.join(OUT_DIR, "ari_results.csv")
RESULT_COLUMNS = [
    "dataset",
    "partition",
    "partition_name",
    "kernel",
    "timestamp",
    "n",
    "features",
    "n_clusters",
    "shape",
    "density",
    "dim_level",
    "size_level",
    "condition",
    "source",
    "ari_mean",
    "ari_std",
    "ari_min",
    "ari_max",
    "nmi_mean",
    "nmi_std",
    "fit_time_s",
    "transform_time_s",
    "kernel_time_s",
    "total_time_s",
    "fit_time_std",
    "ari_per_sec",
    "nmi_per_sec",
    "phi_width",
    "n_estimators",
    "max_samples",
    "n_runs",
]


def _run_clustering_task(task):
    """Worker: run one (dataset, partition, kernel) combination."""
    ds_name, partition, kernel, n_est, max_samples = task
    from data.datasets import DATASETS

    ds = DATASETS[ds_name]
    return run_one(ds, partition, kernel, n_est, max_samples)


def _load_existing(path, n_estimators, partition_max_samples):
    """Load existing CSV and return completed keys for current estimator/psi config."""
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_csv(path)
        # Keep only rows matching requested estimator and per-partition max_samples.
        df_match = df[df["n_estimators"] == n_estimators]
        df_match = df_match[df_match["partition"].isin(partition_max_samples.keys())]
        df_match = df_match[
            df_match["max_samples"] == df_match["partition"].map(partition_max_samples)
        ]
        keys = set(
            zip(
                df_match["dataset"],
                df_match["partition"],
                df_match["kernel"],
                df_match["n_estimators"],
                df_match["max_samples"],
            )
        )
        return keys, df_match.to_dict("records")
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
    n_clusters = len(np.unique(y))

    # Subsample very large datasets
    if len(X) > MAX_N:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[idx], y[idx]

    aris, nmis = [], []
    fit_times, transform_times, kernel_times = [], [], []
    phi_widths = []

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

        # ── transform (phi) ──────────────────────────────────────
        t0 = time.perf_counter()
        phi = part.transform(X)
        transform_t = time.perf_counter() - t0
        phi_widths.append(phi.shape[1])

        # ── kernel matrix ─────────────────────────────────────────
        t0 = time.perf_counter()
        if kernel == "ik":
            K = part.similarity_ik(X)
        else:
            K = part.similarity_idk(X)
        kernel_t = time.perf_counter() - t0

        # ── cluster ───────────────────────────────────────────────
        K = np.clip((K + K.T) / 2, 0, 1)
        np.fill_diagonal(K, 1.0)

        labels = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=RANDOM_STATE + run,
            n_init=10,
        ).fit_predict(K)

        aris.append(adjusted_rand_score(y, labels))
        nmis.append(normalized_mutual_info_score(y, labels))
        fit_times.append(fit_t)
        transform_times.append(transform_t)
        kernel_times.append(kernel_t)

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
        "n_clusters": n_clusters,
        "shape": ds["shape"],
        "density": ds["density"],
        "dim_level": ds["dim_level"],
        "size_level": ds["size_level"],
        "condition": ds["condition"],
        "source": ds["source"],
        # ── performance metrics ───────────────────────────────────
        "ari_mean": round(float(np.mean(aris)), 4),
        "ari_std": round(float(np.std(aris)), 4),
        "ari_min": round(float(np.min(aris)), 4),
        "ari_max": round(float(np.max(aris)), 4),
        "nmi_mean": round(float(np.mean(nmis)), 4),
        "nmi_std": round(float(np.std(nmis)), 4),
        # ── timing breakdown (mean over runs) ────────────────────
        "fit_time_s": round(float(np.mean(fit_times)), 4),
        "transform_time_s": round(float(np.mean(transform_times)), 4),
        "kernel_time_s": round(float(np.mean(kernel_times)), 4),
        "total_time_s": round(
            float(
                np.mean(
                    [
                        f + t + k
                        for f, t, k in zip(fit_times, transform_times, kernel_times)
                    ]
                )
            ),
            4,
        ),
        "fit_time_std": round(float(np.std(fit_times)), 4),
        # ── efficiency metrics ────────────────────────────────────
        "ari_per_sec": round(
            float(np.mean(aris))
            / (
                float(
                    np.mean(
                        [
                            f + t + k
                            for f, t, k in zip(fit_times, transform_times, kernel_times)
                        ]
                    )
                )
                + 1e-9
            ),
            4,
        ),
        "nmi_per_sec": round(
            float(np.mean(nmis))
            / (
                float(
                    np.mean(
                        [
                            f + t + k
                            for f, t, k in zip(fit_times, transform_times, kernel_times)
                        ]
                    )
                )
                + 1e-9
            ),
            4,
        ),
        # ── partition characteristics ─────────────────────────────
        "phi_width": int(np.mean(phi_widths)),  # avg cells across runs
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "n_runs": N_RUNS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--t",
        "--n-estimators",
        dest="t",
        type=int,
        default=None,
        help="Number of estimators to use (defaults to the script constant, or 50 with --fast)",
    )
    parser.add_argument(
        "--psi",
        "--max-samples",
        dest="psi",
        type=int,
        default=None,
        help="Maximum samples per estimator (defaults to the script constant)",
    )
    parser.add_argument(
        "--partition",
        nargs="+",
        default=None,
        help="One or more partitions (space-separated and/or comma-separated)",
    )
    args = parser.parse_args()

    n_est = args.t if args.t is not None else (50 if args.fast else N_ESTIMATORS)
    os.makedirs(OUT_DIR, exist_ok=True)

    cl_datasets = []
    for ds in DATASETS.values():
        if ds["task"] != "C" or (
            args.dataset is not None and ds["name"] != args.dataset
        ):
            continue
        try:
            _ = ds["X"]  # trigger lazy load once
            cl_datasets.append(ds)
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

    partition_max_samples = {
        p: (args.psi if args.psi is not None else DEFAULT_MAX_SAMPLES_BY_PARTITION[p])
        for p in partitions
    }

    print("=" * 68)
    print("  IK Partitioning Study — Clustering Experiments")
    print("=" * 68)
    print(f"  Datasets   : {len(cl_datasets)}")
    print(f"  Partitions : {partitions}")
    print(f"  Kernels    : {KERNELS}")
    if args.psi is not None:
        print(f"  n_est      : {n_est}   psi (override): {args.psi}")
    else:
        psi_desc = ", ".join(f"{p}={partition_max_samples[p]}" for p in partitions)
        print(f"  n_est      : {n_est}   psi defaults: {psi_desc}")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  Output     : {OUT_DIR}")
    print("=" * 68)
    print()

    completed, results = _load_existing(OUT_PATH, n_est, partition_max_samples)
    total = len(cl_datasets) * len(partitions) * len(KERNELS)
    done = len(results)
    skipped = 0

    tasks = []
    for ds in cl_datasets:
        for partition in partitions:
            partition_psi = partition_max_samples[partition]
            for kernel in KERNELS:
                key = (ds["name"], partition, kernel, n_est, partition_psi)
                if key in completed:
                    skipped += 1
                    continue
                tasks.append((ds["name"], partition, kernel, n_est, partition_psi))

    total = len(tasks) + skipped
    done = skipped
    workers = max(1, min(8, len(tasks)))  # cap threads, but always at least 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_clustering_task, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            row = future.result()
            if row is None:
                continue
            results.append(row)
            _append_row(row, OUT_PATH)
            print(
                f"  [{done:3d}/{total}] {row['dataset']:28s} {row['partition']:12s} {row['kernel']}  "
                f"ARI={row['ari_mean']:.3f}+/-{row['ari_std']:.3f}  "
                f"NMI={row['nmi_mean']:.3f}  "
                f"phi={row['phi_width']}  "
                f"t={row['total_time_s']:.1f}s"
            )

    if skipped:
        print(f"\n  Skipped {skipped} already-completed combinations.")

    df = pd.DataFrame(results)
    print(f"\n  Saved {len(df)} rows -> {OUT_PATH}")

    if len(df) == 0:
        return

    print("\n  Mean ARI per partition x kernel:")
    print(
        df.groupby(["partition", "kernel"])["ari_mean"]
        .mean()
        .unstack()
        .round(3)
        .to_string()
    )
    print("\n  Mean NMI per partition x kernel:")
    print(
        df.groupby(["partition", "kernel"])["nmi_mean"]
        .mean()
        .unstack()
        .round(3)
        .to_string()
    )
    print("\n  Mean total_time_s per partition:")
    print(df.groupby("partition")["total_time_s"].mean().round(3).to_string())
    print("\n  Mean ARI per condition x partition:")
    print(
        df.groupby(["condition", "partition"])["ari_mean"]
        .mean()
        .unstack()
        .round(3)
        .to_string()
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
