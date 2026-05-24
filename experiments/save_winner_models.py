"""
experiments/save_winner_models.py
=================================
After experiments finish, this script:
  1. Reads auc_results.csv and ari_results.csv
  2. Finds the winning partition per dataset
  3. Fits ONE model with random_state=42 (matches dashboard)
  4. Saves it as a pickle in results/models/

Usage:
    python experiments/save_winner_models.py
"""

import os, sys, joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from data.datasets import DATASETS
from src.partitions import get_partition, PARTITION_NAMES


def _winners(csv_path, metric):
    """Return {dataset: (partition, kernel, score)} from a results CSV."""
    if not os.path.exists(csv_path):
        print(f"  SKIP {csv_path} not found")
        return {}
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").drop_duplicates(
        ["dataset", "partition", "kernel"], keep="last"
    )
    best = df.loc[df.groupby("dataset")[metric].idxmax()]
    return {
        r["dataset"]: (r["partition"], r["kernel"], r[metric])
        for _, r in best.iterrows()
    }


def main():
    ad_path = os.path.join(ROOT, "results", "anomaly_detection", "auc_results.csv")
    cl_path = os.path.join(ROOT, "results", "clustering", "ari_results.csv")
    out_dir = os.path.join(ROOT, "results", "models")
    os.makedirs(out_dir, exist_ok=True)

    ad_win = _winners(ad_path, "auc_mean")
    cl_win = _winners(cl_path, "ari_mean")

    all_winners = {}
    for ds_name, (part, kern, score) in ad_win.items():
        all_winners.setdefault(ds_name, {})["ad"] = (part, kern, score)
    for ds_name, (part, kern, score) in cl_win.items():
        all_winners.setdefault(ds_name, {})["cl"] = (part, kern, score)

    print("=" * 60)
    print("  Saving winner models")
    print("=" * 60)

    for ds_name, tasks in sorted(all_winners.items()):
        ds = DATASETS.get(ds_name)
        if ds is None:
            print(f"  SKIP {ds_name} — not in DATASETS")
            continue
        X = ds["X"].astype(np.float32)

        for task_key, (part, kern, score) in tasks.items():
            tag = "AD" if task_key == "ad" else "CL"
            pkl = os.path.join(out_dir, f"{ds_name}_{tag}_{part}.pkl")
            if os.path.exists(pkl):
                print(f"  EXIST {pkl}")
                continue

            print(f"  FIT   {ds_name:28s} {tag}  {part:20s}  {score:.3f}")
            part_obj = get_partition(
                part, kernel=kern, n_estimators=200, max_samples=16, random_state=42
            )
            part_obj.fit(X)
            joblib.dump(part_obj, pkl)

    print(f"\n  Done. Models saved to {out_dir}")


if __name__ == "__main__":
    main()
