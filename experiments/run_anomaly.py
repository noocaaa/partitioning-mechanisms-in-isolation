"""
experiments/run_anomaly.py
===========================
Anomaly detection experiments: 4 partitions x IK + IDK x all AD datasets.

Saves to: results/anomaly_detection/auc_results.csv

Usage:
    python experiments/run_anomaly.py
    python experiments/run_anomaly.py --fast
    python experiments/run_anomaly.py --t 200 --psi 256
    python experiments/run_anomaly.py --t 200 --psi 16 --r 3
    python experiments/run_anomaly.py --t 200 --psi 16 --k 3
    python experiments/run_anomaly.py --dataset thyroid
    python experiments/run_anomaly.py --partition anne
    python experiments/run_anomaly.py --partition anne inne
    python experiments/run_anomaly.py --partition anne,inne,iforest
"""

import os, sys, time, argparse, warnings, traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

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
DEFAULT_MAX_SAMPLES_BY_PARTITION = {
    "anne": 16,
    "inne": 16,
    "inne-overlapping": 16,
    "iforest": 256,
    "sciforest": 256,
}
RANDOM_STATE = 42
N_RUNS = 5
MAX_N = 10000  # subsample threshold
DEFAULT_INNE_R = 1.0
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
    "coverage_rate",
    "phi_ones_per_point_per_estimator",
    "phi_ones_per_point_per_estimator_normal",
    "phi_ones_per_point_per_estimator_anomaly",
    "n_estimators",
    "max_samples",
    "r",
    "k",
    "weighting",
    "n_runs",
]

INNE_METHODS = {"inne", "inne-overlapping"}
WEIGHTING_METHODS = {"anne", "inne-overlapping"}
VALID_WEIGHTING = {"binary", "boundary"}


def _run_anomaly_task(task):
    """Worker: run one (dataset, partition, kernel) combination."""
    (
        ds_name,
        partition,
        kernel,
        n_est,
        max_samples,
        r_value,
        k_value,
        weighting_value,
    ) = task
    from data.datasets import DATASETS

    ds = DATASETS[ds_name]
    return run_one(
        ds,
        partition,
        kernel,
        n_est,
        max_samples,
        r_value,
        k_value,
        weighting_value,
    )


def _normalize_r_value(value):
    if pd.isna(value):
        return ""
    if value == "":
        return ""
    return float(value)


def _normalize_k_value(value):
    if pd.isna(value):
        return ""
    if value == "":
        return ""
    return int(value)


def _normalize_existing_k(row):
    if row.get("partition") == "anne":
        value = row.get("k", "")
        return 1 if value == "" or pd.isna(value) else int(value)
    return ""


def _normalize_existing_r(row):
    partition = row.get("partition")
    value = row.get("r", "")
    if partition in INNE_METHODS:
        if value == "" or pd.isna(value):
            return DEFAULT_INNE_R
        return float(value)
    return ""


def _effective_weighting(partition, k_value, weighting_value):
    if partition not in WEIGHTING_METHODS:
        return ""
    if partition == "anne" and int(k_value) == 1:
        return ""
    return str(weighting_value).strip().lower()


def _normalize_weighting_value(value):
    if pd.isna(value):
        return ""
    token = str(value).strip().lower()
    if token in {"", "nan"}:
        return ""
    return token


def _normalize_existing_weighting(row):
    partition = row.get("partition")
    k_value = row.get("k", "")
    value = row.get("weighting", "")
    if value == "":
        return ""
    return _effective_weighting(partition, k_value, value)


def _parse_multi_values(raw_values, cast, arg_name):
    if raw_values is None:
        return None
    tokens = []
    for value in raw_values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    if not tokens:
        raise ValueError(f"No valid values provided for --{arg_name}")

    parsed = []
    for token in tokens:
        try:
            parsed.append(cast(token))
        except Exception as exc:
            raise ValueError(f"Invalid value for --{arg_name}: {token}") from exc

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(parsed))


def _detect_anomaly_label(y):
    labels = np.unique(y)
    if len(labels) == 2:
        counts = np.bincount(y.astype(int))
        anomaly_label = int(np.argmin(counts))  # rarer class = anomaly
    else:
        anomaly_label = 1  # fallback for non-binary (won't compute AUC anyway)
    return anomaly_label, y == anomaly_label


def _load_existing(path, n_estimators, partitions):
    """Load existing CSV and return completed keys for current estimator/partition set."""
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_csv(path)
        if "r" not in df.columns:
            df["r"] = ""
        if "k" not in df.columns:
            df["k"] = ""
        if "weighting" not in df.columns:
            df["weighting"] = ""
        df["r"] = df["r"].apply(_normalize_r_value)
        df["r"] = df.apply(_normalize_existing_r, axis=1)
        df["k"] = df["k"].apply(_normalize_k_value)
        df["k"] = df.apply(_normalize_existing_k, axis=1)
        df["weighting"] = df["weighting"].apply(_normalize_weighting_value)
        df["weighting"] = df.apply(_normalize_existing_weighting, axis=1)
        # Keep only rows matching requested estimator and selected partitions.
        df_match = df[df["n_estimators"] == n_estimators]
        df_match = df_match[df_match["partition"].isin(partitions)]
        keys = set(
            zip(
                df_match["dataset"],
                df_match["partition"],
                df_match["kernel"],
                df_match["n_estimators"],
                df_match["max_samples"],
                df_match["r"],
                df_match["k"],
                df_match["weighting"],
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


def run_one(
    ds,
    partition_method,
    kernel,
    n_estimators,
    max_samples,
    r_value,
    k_value,
    weighting_value,
):
    X = ds["X"].astype(np.float32)
    y = ds["y"]

    if len(X) > MAX_N:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), MAX_N, replace=False)
        X, y = X[idx], y[idx]

    anomaly_label, anomaly_mask = _detect_anomaly_label(y)
    normal_mask = ~anomaly_mask

    aucs = []
    fit_times, transform_times, score_times = [], [], []
    phi_widths = []
    coverage_rates = []
    phi_ones_per_point_per_estimator = []
    phi_ones_per_point_per_estimator_normal = []
    phi_ones_per_point_per_estimator_anomaly = []

    for run in range(N_RUNS):
        effective_weighting = _effective_weighting(
            partition_method, k_value, weighting_value
        )
        part = get_partition(
            partition_method,
            kernel=kernel,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=RANDOM_STATE + run,
            **({"k": k_value} if partition_method == "anne" else {}),
            **({"r": r_value} if partition_method in INNE_METHODS else {}),
            **({"weighting": effective_weighting} if effective_weighting else {}),
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
        coverage_rates.append(part.average_coverage_rate(X, phi=phi, t=n_estimators))
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
            # IDK score: 1 - similarity to training distribution
            scores = part.idk_scores(X)
        score_t = time.perf_counter() - t0

        try:
            labels = np.unique(y)
            if len(labels) == 2:
                y_auc = (y == anomaly_label).astype(int)
                auc = roc_auc_score(y_auc, scores)
            else:
                auc = float("nan")
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
        "anom_rate": round(float(np.mean(anomaly_mask)) * 100.0, 2),
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
        "coverage_rate": round(float(np.nanmean(coverage_rates)), 4),
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
        "r": r_value if partition_method in INNE_METHODS else "",
        "k": k_value if partition_method == "anne" else "",
        "weighting": _effective_weighting(partition_method, k_value, weighting_value),
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
        nargs="+",
        type=str,
        default=None,
        help="One or more max_samples values (space/comma-separated)",
    )
    parser.add_argument(
        "--partition",
        nargs="+",
        default=None,
        help="One or more partitions (space-separated and/or comma-separated)",
    )
    parser.add_argument(
        "--r",
        nargs="+",
        type=str,
        default=None,
        help="One or more radius multipliers for iNNE/iNNE-overlapping (space/comma-separated)",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=str,
        default=None,
        help="One or more nearest-centroid values for aNNE (anne only; defaults to 1)",
    )
    parser.add_argument(
        "--weighting",
        nargs="+",
        type=str,
        default=None,
        help="Weighting mode(s) for anne/inne/inne-overlapping: binary or boundary",
    )
    args = parser.parse_args()

    n_est = args.t if args.t is not None else (50 if args.fast else N_ESTIMATORS)
    os.makedirs(OUT_DIR, exist_ok=True)

    ad_datasets = []
    for ds in DATASETS.values():
        if ds["task"] != "AD" or (
            args.dataset is not None and ds["name"] != args.dataset
        ):
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

    try:
        psi_values = _parse_multi_values(args.psi, int, "psi")
        r_values = _parse_multi_values(args.r, float, "r")
        k_values = _parse_multi_values(args.k, int, "k")
        weighting_values = _parse_multi_values(args.weighting, str, "weighting")
    except ValueError as exc:
        parser.error(str(exc))

    if k_values is None:
        k_values = [1]

    if weighting_values is None:
        weighting_values = ["boundary"]
    else:
        weighting_values = [str(v).strip().lower() for v in weighting_values]
        invalid_weighting = sorted(set(weighting_values) - VALID_WEIGHTING)
        if invalid_weighting:
            parser.error(
                f"Unknown weighting value(s): {', '.join(invalid_weighting)}. "
                f"Valid options: {', '.join(sorted(VALID_WEIGHTING))}"
            )

    partition_param_grid = {}
    for p in partitions:
        psi_list = (
            psi_values
            if psi_values is not None
            else [DEFAULT_MAX_SAMPLES_BY_PARTITION[p]]
        )
        r_list = r_values if (p in INNE_METHODS and r_values is not None) else [""]
        k_list = k_values if p == "anne" else [""]
        if p == "anne":
            combos = []
            for psi, r, k in product(psi_list, r_list, k_list):
                local_weighting = [""] if int(k) == 1 else weighting_values
                for w in local_weighting:
                    combos.append((psi, r, k, w))
            partition_param_grid[p] = combos
        else:
            if p in INNE_METHODS:
                r_list = r_values if r_values is not None else [DEFAULT_INNE_R]
            weighting_list = weighting_values if p in WEIGHTING_METHODS else [""]
            partition_param_grid[p] = list(
                product(psi_list, r_list, k_list, weighting_list)
            )

    print("=" * 68)
    print("  IK Partitioning Study — Anomaly Detection Experiments")
    print("=" * 68)
    print(f"  Datasets   : {len(ad_datasets)}")
    print(f"  Partitions : {partitions}")
    print(f"  Kernels    : {KERNELS}")
    if psi_values is not None:
        print(f"  n_est      : {n_est}   psi values: {psi_values}")
    else:
        psi_desc = ", ".join(
            f"{p}={DEFAULT_MAX_SAMPLES_BY_PARTITION[p]}" for p in partitions
        )
        print(f"  n_est      : {n_est}   psi defaults: {psi_desc}")
    if r_values is not None:
        print(f"  r (iNNE)   : {r_values}")
    if "anne" in partitions:
        print(f"  k (aNNE)   : {k_values}")
    if any(p in WEIGHTING_METHODS for p in partitions):
        print(f"  weighting  : {weighting_values}")
    print(f"  Runs each  : {N_RUNS}")
    print(f"  Output     : {OUT_DIR}")
    print("=" * 68)
    print()

    completed, results = _load_existing(OUT_PATH, n_est, partitions)
    total = len(ad_datasets) * len(partitions) * len(KERNELS)
    done = len(results)
    skipped = 0

    tasks = []
    for ds in ad_datasets:
        for partition in partitions:
            for (
                partition_psi,
                partition_r,
                partition_k,
                partition_weighting,
            ) in partition_param_grid[partition]:
                for kernel in KERNELS:
                    key = (
                        ds["name"],
                        partition,
                        kernel,
                        n_est,
                        partition_psi,
                        partition_r,
                        partition_k,
                        partition_weighting,
                    )
                    if key in completed:
                        skipped += 1
                        continue
                    tasks.append(
                        (
                            ds["name"],
                            partition,
                            kernel,
                            n_est,
                            partition_psi,
                            partition_r,
                            partition_k,
                            partition_weighting,
                        )
                    )

    total = len(tasks) + skipped
    done = skipped
    workers = max(1, min(8, len(tasks)))  # cap threads, but always at least 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_anomaly_task, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                row = future.result()
            except Exception as exc:
                task = futures[future]
                print(f"  [ERROR] task {task} raised: {exc}")
                traceback.print_exc()
                continue
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
