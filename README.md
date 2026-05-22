# IK Partition Study

Comparative study of four partitioning mechanisms used in **IsoKernel (IK)** and **Isolation Distributional Kernel (IDK)** methods for clustering and anomaly detection.

---

## Overview

This repository evaluates four partitioning strategies across **41 datasets** organized into **7 experimental conditions** (e.g., spherical, elongated, crescent, nested, varying density, high-dimensional, large). Each condition is designed to reveal strengths and weaknesses of specific partitioning mechanisms based on prior literature (Cao et al. 2025).

**Tasks covered:**

- **Clustering (C)** — 23 datasets, evaluated with ARI and NMI
- **Anomaly Detection (AD)** — 18 datasets, evaluated with AUC

---

## Partitioning Mechanisms

| Short name  | Full name                     | Geometry                                              |
| ----------- | ----------------------------- | ----------------------------------------------------- |
| `anne`      | Voronoi (aNNE)                | Voronoi cells — nearest centroid assignment           |
| `inne`      | Hypersphere (iNNE)            | Hyperspheres — radius = NN distance of centroid       |
| `iforest`   | Axis-parallel (iForest)       | Hyper-rectangles — axis-aligned recursive splits      |
| `sciforest` | Random hyperplane (SCiForest) | Oblique partitions — random linear combination splits |

---

## Datasets

All datasets are loaded centrally via `data/datasets.py` into a global dict `DATASETS`.

**Sources:**

- `sklearn` — Built-in datasets (iris, wine, digits, breast_cancer)
- `sklearn_gen` — Synthetic generators (blobs, moons, circles, etc.)
- `UCI` — Fetched via `ucimlrepo` package, requires internet
- `ADBench` — Auto-downloaded from GitHub (`Minqi824/ADBench`) on first run, saved to `data/anomaly_detection/`

**First-run behavior:** When `data/datasets.py` runs for the first time, missing ADBench `.npz` files are downloaded automatically. If downloads fail, datasets are skipped gracefully.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** The ADBench auto-downloader temporarily disables SSL hostname verification to avoid certificate errors on some networks.

---

## Usage

### Run experiments

```bash
# Full anomaly detection experiments (4 partitions × 2 kernels × 18 AD datasets)
python experiments/run_anomaly.py

# Full clustering experiments (4 partitions × 2 kernels × 23 C datasets)
python experiments/run_clustering.py

# Fast mode (n_estimators=50 instead of 200)
python experiments/run_anomaly.py --fast
python experiments/run_clustering.py --fast

# Run single dataset or partition
python experiments/run_anomaly.py --dataset thyroid --partition anne
python experiments/run_clustering.py --dataset iris --partition inne

# Run multiple partitions in one go
python experiments/run_anomaly.py --dataset thyroid --partition anne inne
python experiments/run_clustering.py --dataset iris --partition anne,iforest
```

### Run dashboards

```bash
# EDA dashboard — explore dataset shapes and class distributions
python notebooks/eda.py
# Opens at http://127.0.0.1:8053

# Partition visualizer — explore partitions interactively
python notebooks/visualize_dashboard.py
# Opens at http://127.0.0.1:8054
```

### Module sanity checks

```bash
python src/partitions.py      # Validates all 4 partitions × IK + IDK
python data/datasets.py       # Prints dataset summary table
```

---

## Project Structure

```
├── data/
│   ├── datasets.py              # Central data loader (all 41 datasets)
│   ├── anomaly_detection/       # ADBench .npz files (auto-downloaded)
│   └── README.md                # Detailed dataset documentation
├── src/
│   ├── partitions.py            # Core: 4 partition implementations + factory
│   ├── evaluation.py            # Metrics: AUC, ARI, NMI, timing helper
│   ├── preprocessing.py         # Data loading helpers
│   └── visualize.py             # 2D geometry plots, kernel heatmaps, phi maps
├── experiments/
│   ├── run_anomaly.py           # AD experiments
│   ├── run_clustering.py        # Clustering experiments
│   └── run_synthetic.py         # Synthetic experiments (placeholder)
├── notebooks/
│   ├── eda.py                   # Interactive Dash/Plotly EDA dashboard
│   ├── visualize_dashboard.py   # Interactive partition visualizer
│   └── visualize_partitions.py  # Static partition visualizations
├── results/
│   ├── anomaly_detection/       # Output CSVs (e.g., auc_results.csv)
│   ├── clustering/              # Output CSVs (e.g., ari_results.csv)
│   └── figures/                 # Generated figures
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Results

Experiment outputs are saved to:

- `results/anomaly_detection/auc_results.csv` — AUC scores per dataset × partition × kernel
- `results/clustering/ari_results.csv` — ARI scores per dataset × partition × kernel
- `results/clustering/nmi_results.csv` — NMI scores per dataset × partition × kernel

Each CSV includes timing breakdowns (fit, transform, score) and partition characteristics (φ width, n_estimators, max_samples).

---

## Team Members

- Noelia Carrasco
- Rafał Michal Kaminski
