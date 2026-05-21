# Data

This folder contains all datasets used in the IK Partitioning Study.

## Structure

```
data/
├── datasets.py              ← single entry point: run this to load everything
├── anomaly_detection/       ← ODDS datasets (.npz, auto-downloaded on first run)
└── clustering/              ← UCI datasets (.csv, downloaded via ucimlrepo)
```

## How to load datasets

```python
from data.datasets import DATASETS, print_summary

# All 41 datasets as a dict
print_summary()   # prints the full summary table

# Access one dataset
ds = DATASETS['iris']
X  = ds['X']          # np.ndarray, shape (n, features), normalized to [0,1]
y  = ds['y']          # np.ndarray, shape (n,), integer labels
task = ds['task']     # 'C' (clustering) or 'AD' (anomaly detection)
cond = ds['condition']# 1–7 (see conditions below)

# Filter by task
clustering_sets = {k: v for k, v in DATASETS.items() if v['task'] == 'C'}
anomaly_sets    = {k: v for k, v in DATASETS.items() if v['task'] == 'AD'}

# Filter by condition
cond3_sets = {k: v for k, v in DATASETS.items() if v['condition'] == 3}
```

## Dataset fields

| Field | Type | Description |
|---|---|---|
| `name` | str | Dataset identifier |
| `X` | np.ndarray | Feature matrix, normalized to [0,1] |
| `y` | np.ndarray | Ground truth labels (integers) |
| `task` | str | `'C'` = clustering, `'AD'` = anomaly detection |
| `n` | int | Number of samples |
| `features` | int | Number of features |
| `shape` | str | Cluster geometry |
| `density` | str | `'uniform'`, `'varying'`, or `'sparse'` |
| `dim_level` | str | `'low'`, `'mid'`, `'high'`, `'vhigh'` |
| `size_level` | str | `'small'`, `'medium'`, `'large'` |
| `source` | str | `'sklearn'`, `'sklearn_gen'`, `'UCI'`, `'ADBench'` |
| `condition` | int | Experimental condition (1–7) |

## Experimental conditions

The 7 conditions are designed to reveal strengths and weaknesses of each partitioning mechanism:

| # | Condition | Why it matters |
|---|---|---|
| 1 | **Spherical** | Baseline — all 4 partitions should perform similarly |
| 2 | **Elongated / elliptical** | iForest (axis-parallel) struggles; random hyperplane wins |
| 3 | **Crescent / irregular** | iForest FAILS (confirmed in survey paper); iNNE wins |
| 4 | **Nested / concentric** | Hardest case for all hyperplane-based partitions |
| 5 | **Varying density** | iNNE adapts ball size to density; iForest does not |
| 6 | **High-dimensional** | Voronoi degrades; random hyperplane handles it better |
| 7 | **Large (efficiency)** | iForest is fastest; reveals computational cost differences |

## Sources

| Source | How datasets are loaded | Download required? |
|---|---|---|
| `sklearn` | sklearn built-in (`load_iris`, etc.) | No |
| `sklearn_gen` | sklearn generators (`make_blobs`, etc.) | No |
| `UCI` | `ucimlrepo` Python package | Internet connection |
| `ADBench` | Auto-downloaded from GitHub on first run | Internet (first run only) |

### First run behaviour

When `datasets.py` runs for the first time, ADBench `.npz` files are downloaded automatically from:

```
https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/
```

Files are saved to `data/anomaly_detection/` and reused on subsequent runs.

### ADBench filename mapping

| Dataset name | ADBench file |
|---|---|
| annthyroid | `2_annthyroid.npz` |
| breastw | `4_breastw.npz` |
| cardio | `6_cardio.npz` |
| ionosphere | `18_Ionosphere.npz` |
| letter | `20_letter.npz` |
| lympho | `21_Lymphography.npz` |
| musk | `25_musk.npz` |
| optdigits | `26_optdigits.npz` |
| pendigits | `28_pendigits.npz` |
| satellite | `30_satellite.npz` |
| shuttle | `32_shuttle.npz` |
| thyroid | `38_thyroid.npz` |
| vowels | `40_vowels.npz` |
| waveform | `41_Waveform.npz` |
| wbc | `42_WBC.npz` |
| yeast_ad | `47_yeast.npz` |

## Dataset summary

| Condition | Datasets | C | AD |
|---|---|---|---|
| 1 — Spherical | 7 | 3 | 4 |
| 2 — Elongated | 6 | 5 | 1 |
| 3 — Crescent | 5 | 2 | 3 |
| 4 — Nested | 3 | 3 | 0 |
| 5 — Varying density | 7 | 5 | 2 |
| 6 — High-dimensional | 8 | 4 | 4 |
| 7 — Large | 5 | 1 | 4 |
| **Total** | **41** | **23** | **18** |