"""
src/analysis.py  —  Results analysis & roadmap generator
=======================================================
Run after experiments to get a printable summary of:
  • Best partition per condition (clustering vs anomaly detection)
  • Cost / performance trade-offs
  • Unified cross-task recommendations

Usage:
    python src/analysis.py
    python src/analysis.py --csv results/anomaly_detection/auc_results.csv
    python src/analysis.py --csv results/clustering/ari_results.csv
"""

import os, sys, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.datasets import DATASETS

COND_NAMES = {
    1: "Spherical (baseline)",
    2: "Elongated / elliptical",
    3: "Crescent / irregular",
    4: "Nested / concentric",
    5: "Varying density",
    6: "High-dimensional",
    7: "Large (efficiency)",
}

PARTITION_NAMES = {
    'anne':     'Voronoi (aNNE)',
    'inne':     'Hypersphere (iNNE)',
    'inne-overlapping': 'Hypersphere (iNNE) Overlapping',
    'iforest':  'Axis-parallel (iForest)',
    'sciforest':'Random hyperplane (SCiForest)',
}


def _load_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Normalize partition names
    df['partition_name'] = df['partition'].map(PARTITION_NAMES)
    return df


def _summarise_task(df, metric, task_label):
    """Print per-condition summary for one task."""
    print(f"\n{'='*72}")
    print(f"  {task_label} — Mean {metric.upper()} per condition × partition")
    print(f"{'='*72}")

    pivot = df.groupby(['condition', 'partition'])[f'{metric}_mean'].mean().unstack()
    pivot = pivot.reindex(columns=['anne', 'inne', 'inne-overlapping', 'iforest', 'sciforest'])
    pivot.columns = [PARTITION_NAMES[c] for c in pivot.columns]
    pivot.index = [f"C{i} — {COND_NAMES[i]}" for i in pivot.index]
    print(pivot.round(3).to_string())

    print(f"\n  🏆 Best partition per condition ({task_label}):")
    for cond in sorted(df['condition'].unique()):
        sub = df[df['condition'] == cond]
        best = sub.groupby('partition')[f'{metric}_mean'].mean().idxmax()
        score = sub.groupby('partition')[f'{metric}_mean'].mean().max()
        print(f"    C{cond} ({COND_NAMES[cond]:22s}) → {PARTITION_NAMES[best]:25s}  {metric.upper()}={score:.3f}")


def _efficiency_summary(ad_df, cl_df):
    """Print cost / performance trade-off tables."""
    print(f"\n{'='*72}")
    print("  COST / PERFORMANCE TRADE-OFFS")
    print(f"{'='*72}")

    # --- Anomaly detection ---
    if ad_df is not None:
        print("\n  Anomaly Detection — AUC per second (higher = more bang for the buck)")
        eff = ad_df.groupby('partition')[['auc_per_sec', 'total_time_s']].mean()
        eff = eff.reindex(['anne', 'inne', 'inne-overlapping', 'iforest', 'sciforest'])
        eff.index = [PARTITION_NAMES[i] for i in eff.index]
        print(eff.round(3).to_string())

    # --- Clustering ---
    if cl_df is not None:
        print("\n  Clustering — ARI per second (higher = more bang for the buck)")
        eff = cl_df.groupby('partition')[['ari_per_sec', 'total_time_s']].mean()
        eff = eff.reindex(['anne', 'inne', 'inne-overlapping', 'iforest', 'sciforest'])
        eff.index = [PARTITION_NAMES[i] for i in eff.index]
        print(eff.round(3).to_string())


def _cross_task_roadmap(ad_df, cl_df):
    """Build a unified roadmap table."""
    print(f"\n{'='*72}")
    print("  UNIFIED ROADMAP — When to use which partition?")
    print(f"{'='*72}")

    rows = []
    for cond in range(1, 8):
        row = {'condition': f"C{cond}", 'scenario': COND_NAMES[cond]}

        if ad_df is not None:
            ad_sub = ad_df[ad_df['condition'] == cond]
            if not ad_sub.empty:
                ad_best = ad_sub.groupby('partition')['auc_mean'].mean().idxmax()
                row['AD_best'] = PARTITION_NAMES[ad_best]
                row['AD_auc'] = round(ad_sub.groupby('partition')['auc_mean'].mean().max(), 3)
            else:
                row['AD_best'] = '—'
                row['AD_auc'] = '—'

        if cl_df is not None:
            cl_sub = cl_df[cl_df['condition'] == cond]
            if not cl_sub.empty:
                cl_best = cl_sub.groupby('partition')['ari_mean'].mean().idxmax()
                row['CL_best'] = PARTITION_NAMES[cl_best]
                row['CL_ari'] = round(cl_sub.groupby('partition')['ari_mean'].mean().max(), 3)
            else:
                row['CL_best'] = '—'
                row['CL_ari'] = '—'

        rows.append(row)

    roadmap = pd.DataFrame(rows)
    print("\n  " + roadmap.to_string(index=False))

    # Consensus recommendation
    print(f"\n  📋 Consensus recommendation per condition:")
    for _, r in roadmap.iterrows():
        cond = r['condition']
        scenario = r['scenario']
        ad = r.get('AD_best', '—')
        cl = r.get('CL_best', '—')
        if ad == cl and ad != '—':
            print(f"    {cond} ({scenario:22s}) → {ad:25s}  (wins on BOTH tasks)")
        elif ad != '—' and cl != '—':
            print(f"    {cond} ({scenario:22s}) → AD:{ad:25s} | CL:{cl:25s}")
        elif ad != '—':
            print(f"    {cond} ({scenario:22s}) → AD:{ad:25s} | CL: no data")
        else:
            print(f"    {cond} ({scenario:22s}) → AD: no data | CL:{cl:25s}")


def _dataset_breakdown(ad_df, cl_df):
    """Show best partition per individual dataset."""
    print(f"\n{'='*72}")
    print("  PER-DATASET WINNERS")
    print(f"{'='*72}")

    if ad_df is not None:
        print("\n  Anomaly Detection — best partition per dataset:")
        for ds in sorted(ad_df['dataset'].unique()):
            sub = ad_df[ad_df['dataset'] == ds]
            best = sub.groupby('partition')['auc_mean'].mean().idxmax()
            score = sub.groupby('partition')['auc_mean'].mean().max()
            cond = sub['condition'].iloc[0]
            print(f"    {ds:30s}  C{cond}  → {PARTITION_NAMES[best]:25s}  AUC={score:.3f}")

    if cl_df is not None:
        print("\n  Clustering — best partition per dataset:")
        for ds in sorted(cl_df['dataset'].unique()):
            sub = cl_df[cl_df['dataset'] == ds]
            best = sub.groupby('partition')['ari_mean'].mean().idxmax()
            score = sub.groupby('partition')['ari_mean'].mean().max()
            cond = sub['condition'].iloc[0]
            print(f"    {ds:30s}  C{cond}  → {PARTITION_NAMES[best]:25s}  ARI={score:.3f}")


def _time_summary(ad_df, cl_df):
    """Show raw timing comparison."""
    print(f"\n{'='*72}")
    print("  RAW TIMING COMPARISON (seconds per run)")
    print(f"{'='*72}")

    for df, label in [(ad_df, 'Anomaly Detection'), (cl_df, 'Clustering')]:
        if df is None:
            continue
        print(f"\n  {label}:")
        timing = df.groupby('partition')[['fit_time_s', 'transform_time_s', 'total_time_s']].mean()
        timing = timing.reindex(['anne', 'inne', 'inne-overlapping', 'iforest', 'sciforest'])
        timing.index = [PARTITION_NAMES[i] for i in timing.index]
        print(timing.round(3).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ad', type=str, default=os.path.join(ROOT, 'results', 'anomaly_detection', 'auc_results.csv'))
    parser.add_argument('--cl', type=str, default=os.path.join(ROOT, 'results', 'clustering', 'ari_results.csv'))
    args = parser.parse_args()

    ad_df = _load_csv(args.ad)
    cl_df = _load_csv(args.cl)

    if ad_df is None and cl_df is None:
        print("No results found. Run experiments first:")
        print("  python experiments/run_anomaly.py --fast")
        print("  python experiments/run_clustering.py --fast")
        return

    print(f"\n{'='*72}")
    print("  IK PARTITION STUDY — RESULTS ANALYSIS")
    print(f"{'='*72}")
    if ad_df is not None:
        print(f"  Anomaly detection rows : {len(ad_df)}")
    if cl_df is not None:
        print(f"  Clustering rows        : {len(cl_df)}")

    if ad_df is not None:
        _summarise_task(ad_df, 'auc', 'Anomaly Detection')
    if cl_df is not None:
        _summarise_task(cl_df, 'ari', 'Clustering')

    _cross_task_roadmap(ad_df, cl_df)
    _efficiency_summary(ad_df, cl_df)
    _time_summary(ad_df, cl_df)
    _dataset_breakdown(ad_df, cl_df)

    print(f"\n{'='*72}")
    print("  Done.")
    print(f"{'='*72}\n")


if __name__ == '__main__':
    main()
