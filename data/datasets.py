"""
datasets.py  —  IK Partitioning Study
======================================
Loads all datasets for the study.

Sources
-------
- sklearn builtins/generators : always available, no download
- ucimlrepo                   : pip install ucimlrepo  (needs internet)
- ADBench (ODDS datasets)     : auto-downloaded from GitHub on first run
                                saved to data/anomaly_detection/

Usage
-----
    from data.datasets import DATASETS, print_summary
    print_summary()

    for name, ds in DATASETS.items():
        X, y = ds['X'], ds['y']   # normalized X, integer y
        task = ds['task']         # 'C' or 'AD'
        cond = ds['condition']    # 1-7
"""

import os
import io
import ssl
import warnings
import urllib.request

import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_blobs, make_moons, make_circles,
    make_gaussian_quantiles, make_classification,
)

warnings.filterwarnings('ignore')

# ── ADBench filename map ───────────────────────────────────────────────────
# Maps friendly name → exact filename in ADBench GitHub repo
ADBENCH_FILES = {
    'annthyroid':  '2_annthyroid.npz',
    'breastw':     '4_breastw.npz',
    'cardio':      '6_cardio.npz',
    'ionosphere':  '18_Ionosphere.npz',
    'letter':      '20_letter.npz',
    'lympho':      '21_Lymphography.npz',
    'musk':        '25_musk.npz',
    'optdigits':   '26_optdigits.npz',
    'pendigits':   '28_pendigits.npz',
    'satellite':   '30_satellite.npz',
    'satimage2':   '31_satimage-2.npz',
    'shuttle':     '32_shuttle.npz',
    'thyroid':     '38_thyroid.npz',
    'vowels':      '40_vowels.npz',
    'waveform':    '41_Waveform.npz',
    'wbc':         '42_WBC.npz',
    'yeast_ad':    '47_yeast.npz',
}

ADBENCH_BASE = (
    'https://github.com/Minqi824/ADBench'
    '/raw/main/adbench/datasets/Classical'
)

SAVE_DIR = 'data/anomaly_detection'

# ── Storage ────────────────────────────────────────────────────────────────
DATASETS = {}


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _normalize(X):
    return MinMaxScaler().fit_transform(
        np.nan_to_num(X.astype(float)))


def _encode(y):
    return LabelEncoder().fit_transform(np.array(y).ravel())


def _register(name, X, y, task, shape, density,
               dim_level, size_level, source, condition):
    """Normalize, encode, store and print one dataset."""
    X = _normalize(X)
    y = _encode(y)
    DATASETS[name] = dict(
        name=name, X=X, y=y, task=task,
        n=X.shape[0], features=X.shape[1],
        shape=shape, density=density,
        dim_level=dim_level, size_level=size_level,
        source=source, condition=condition,
    )
    src_tag = '[synth]' if source in ('sklearn_gen',) else '[real ]'
    print(f'  OK   {src_tag} {name:28s} '
          f'n={X.shape[0]:6d}  feat={X.shape[1]:4d}  '
          f'task={task}  cond={condition}')


def _download_adbench(name):
    """
    Download one ADBench dataset from GitHub.
    Returns local path on success, None on failure.
    """
    fname = ADBENCH_FILES.get(name)
    if fname is None:
        return None

    os.makedirs(SAVE_DIR, exist_ok=True)
    dest = os.path.join(SAVE_DIR, fname)

    if os.path.exists(dest):
        return dest   # already downloaded

    url = f'{ADBENCH_BASE}/{fname}'
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req, context=ctx, timeout=20).read()
        np.load(io.BytesIO(data), allow_pickle=True)   # validate
        with open(dest, 'wb') as f:
            f.write(data)
        return dest
    except Exception:
        return None


def _load_adbench(name):
    """
    Load ADBench dataset: local file → download → None.
    Returns (X, y) or (None, None).
    """
    fname = ADBENCH_FILES.get(name)
    if fname is None:
        return None, None

    # local file already there?
    local = os.path.join(SAVE_DIR, fname)
    if not os.path.exists(local):
        print(f'         downloading {name}...', end=' ', flush=True)
        local = _download_adbench(name)
        print('OK' if local else 'FAILED')

    if local and os.path.exists(local):
        d = np.load(local, allow_pickle=True)
        return d['X'].astype(float), d['y'].ravel().astype(int)

    return None, None


def _load_uci(dataset_id):
    """Load a UCI dataset via ucimlrepo. Returns (X, y)."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=dataset_id)
    X = ds.data.features.values.astype(float)
    y = ds.data.targets.values.ravel()
    return X, y


def _try_uci(name, dataset_id, task, shape, density,
             dim_level, size_level, condition):
    """Attempt to load a UCI dataset; skip gracefully on failure."""
    try:
        X, y = _load_uci(dataset_id)
        _register(name, X, y, task, shape, density,
                  dim_level, size_level, 'UCI', condition)
    except Exception as e:
        print(f'  SKIP {name:30s} (ucimlrepo id={dataset_id}): '
              f'{str(e)[:60]}')


def _try_adbench(name, task, shape, density,
                 dim_level, size_level, condition):
    """Attempt to load an ADBench dataset; skip gracefully on failure."""
    X, y = _load_adbench(name)
    if X is not None:
        _register(name, X, y, task, shape, density,
                  dim_level, size_level, 'ADBench', condition)
    else:
        print(f'  SKIP {name:30s} '
              f'(place {ADBENCH_FILES.get(name,"")} '
              f'in {SAVE_DIR}/)')


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 1 — Spherical clusters
# Baseline: all 4 partitions should perform similarly here
# ══════════════════════════════════════════════════════════════════════════
print('\nCondition 1 — Spherical (baseline):')

d = load_iris()
_register('iris', d.data, d.target,
          'C', 'spherical', 'uniform', 'low', 'small', 'sklearn', 1)

d = load_breast_cancer()
_register('breast_cancer', d.data, d.target,
          'AD', 'spherical', 'uniform', 'mid', 'medium', 'sklearn', 1)

X, y = make_blobs(n_samples=500, centers=3,
                  cluster_std=0.8, random_state=42)
_register('syn_blobs_small', X, y,
          'C', 'spherical', 'uniform', 'low', 'small', 'sklearn_gen', 1)

X, y = make_blobs(n_samples=3000, centers=4,
                  cluster_std=1.0, random_state=42)
_register('syn_blobs_medium', X, y,
          'C', 'spherical', 'uniform', 'low', 'medium', 'sklearn_gen', 1)

_try_uci('wbc_uci', 15,
         'AD', 'spherical', 'uniform', 'mid', 'small', 1)

_try_adbench('breastw',
             'AD', 'spherical', 'uniform', 'low', 'small', 1)

_try_adbench('wbc',
             'AD', 'spherical', 'uniform', 'mid', 'small', 1)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 2 — Elongated / elliptical clusters
# iForest (axis-parallel) struggles; random hyperplane should win
# ══════════════════════════════════════════════════════════════════════════
print('\nCondition 2 — Elongated / elliptical:')

d = load_wine()
_register('wine', d.data, d.target,
          'C', 'elliptical', 'uniform', 'low', 'small', 'sklearn', 2)

X, y = make_blobs(n_samples=500, centers=3,
                  cluster_std=[3.0, 0.5, 2.0], random_state=42)
_register('syn_elongated_small', X, y,
          'C', 'elliptical', 'uniform', 'low', 'small', 'sklearn_gen', 2)

X, y = make_blobs(n_samples=800, centers=4,
                  cluster_std=[4.0, 0.3, 2.5, 1.0], random_state=42)
_register('syn_elongated_medium', X, y,
          'C', 'elliptical', 'uniform', 'low', 'medium', 'sklearn_gen', 2)

_try_uci('vehicle', 149,
         'C', 'elliptical', 'uniform', 'mid', 'medium', 2)

_try_uci('glass', 42,
         'C', 'elliptical', 'uniform', 'low', 'small', 2)

_try_adbench('ionosphere',
             'AD', 'elliptical', 'uniform', 'mid', 'small', 2)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 3 — Crescent / chain / irregular
# From survey paper Table 3: iForest FAILS here, iNNE wins
# ══════════════════════════════════════════════════════════════════════════
print('\nCondition 3 — Crescent / irregular:')

X, y = make_moons(n_samples=500, noise=0.05, random_state=42)
_register('syn_moons_small', X, y,
          'C', 'crescent', 'uniform', 'low', 'small', 'sklearn_gen', 3)

X, y = make_moons(n_samples=1000, noise=0.10, random_state=42)
_register('syn_moons_medium', X, y,
          'C', 'crescent', 'uniform', 'low', 'medium', 'sklearn_gen', 3)

# Moons as AD: inliers = moon shapes, outliers = random scatter
X_in, _ = make_moons(n_samples=950, noise=0.05, random_state=42)
rng = np.random.RandomState(0)
X_out = rng.uniform(-2, 3, (50, 2))
X_ad = np.vstack([X_in, X_out])
y_ad = np.array([0]*950 + [1]*50)
_register('syn_moons_ad', X_ad, y_ad,
          'AD', 'crescent', 'uniform', 'low', 'medium', 'sklearn_gen', 3)

_try_adbench('vowels',
             'AD', 'irregular', 'uniform', 'low', 'medium', 3)

_try_adbench('lympho',
             'AD', 'irregular', 'sparse',  'low', 'small',  3)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 4 — Nested / concentric
# Hardest for all hyperplane-based partitions
# ══════════════════════════════════════════════════════════════════════════
print('\nCondition 4 — Nested / concentric:')

X, y = make_circles(n_samples=500, noise=0.05,
                    factor=0.4, random_state=42)
_register('syn_circles_small', X, y,
          'C', 'nested', 'uniform', 'low', 'small', 'sklearn_gen', 4)

X, y = make_circles(n_samples=800, noise=0.08,
                    factor=0.5, random_state=42)
_register('syn_circles_medium', X, y,
          'C', 'nested', 'uniform', 'low', 'medium', 'sklearn_gen', 4)

X, y = make_gaussian_quantiles(n_samples=500, n_features=2,
                               n_classes=3, random_state=42)
_register('syn_gauss_quantiles', X, y,
          'C', 'nested', 'uniform', 'low', 'small', 'sklearn_gen', 4)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 5 — Varying density
# Key test: iNNE adapts ball size to density; iForest does not
# ══════════════════════════════════════════════════════════════════════════
print('\nCondition 5 — Varying density:')

X1, _ = make_blobs(400, centers=[[0, 0]], cluster_std=0.3, random_state=42)
X2, _ = make_blobs(100, centers=[[6, 6]], cluster_std=2.5, random_state=42)
X = np.vstack([X1, X2])
y = np.array([0]*400 + [1]*100)
_register('syn_density_2d', X, y,
          'C', 'mixed', 'varying', 'low', 'small', 'sklearn_gen', 5)

X1, _ = make_blobs(300, centers=[[0,0,0]], cluster_std=0.2, random_state=42)
X2, _ = make_blobs(300, centers=[[5,5,5]], cluster_std=1.5, random_state=42)
X3, _ = make_blobs(400, centers=[[10,0,5]], cluster_std=3.0, random_state=42)
X = np.vstack([X1, X2, X3])
y = np.array([0]*300 + [1]*300 + [2]*400)
_register('syn_density_3d', X, y,
          'C', 'mixed', 'varying', 'low', 'medium', 'sklearn_gen', 5)

_try_uci('ecoli', 39,
         'C', 'mixed', 'varying', 'low', 'small', 5)

_try_uci('yeast', 110,
         'C', 'mixed', 'varying', 'low', 'medium', 5)

_try_adbench('thyroid',
             'AD', 'mixed', 'varying', 'low', 'medium', 5)

_try_adbench('cardio',
             'AD', 'mixed', 'varying', 'mid', 'medium', 5)

_try_adbench('waveform',
             'C',  'mixed', 'varying', 'mid', 'medium', 5)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 6 — High dimensionality
# Voronoi degrades in high-dim; random hyperplane handles it better
# ══════════════════════════════════════════════════════════════════════════
print('\nCondition 6 — High-dimensional:')

d = load_digits()
_register('digits', d.data, d.target,
          'C', 'mixed', 'uniform', 'high', 'large', 'sklearn', 6)

X, y = make_classification(n_samples=1000, n_features=50,
    n_informative=20, n_redundant=10,
    n_classes=3, n_clusters_per_class=1, random_state=42)
_register('syn_highdim_50', X, y,
          'C', 'spherical', 'uniform', 'high', 'medium', 'sklearn_gen', 6)

X, y = make_classification(n_samples=1000, n_features=100,
    n_informative=30, n_redundant=20,
    n_classes=4, n_clusters_per_class=1, random_state=42)
_register('syn_highdim_100', X, y,
          'C', 'spherical', 'uniform', 'vhigh', 'medium', 'sklearn_gen', 6)

_try_uci('dermatology', 33,
         'C', 'mixed', 'uniform', 'high', 'small', 6)

_try_adbench('satellite',
             'AD', 'mixed', 'varying', 'high',  'large',  6)

_try_adbench('musk',
             'AD', 'mixed', 'uniform', 'vhigh', 'medium', 6)

_try_adbench('optdigits',
             'AD', 'mixed', 'uniform', 'high',  'medium', 6)

_try_adbench('pendigits',
             'AD', 'mixed', 'uniform', 'low',   'large',  6)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 7 — Large datasets (computational efficiency)
# iForest is O(n log n) and cheapest; Voronoi most expensive
# ══════════════════════════════════════════════════════════════════════════
print('\nCondition 7 — Large (efficiency):')

X, y = make_blobs(n_samples=10000, centers=5,
                  cluster_std=1.5, random_state=42)
_register('syn_large_10k', X, y,
          'AD', 'spherical', 'uniform', 'low', 'large', 'sklearn_gen', 7)

X, y = make_blobs(n_samples=20000, centers=5,
                  cluster_std=1.5, random_state=42)
_register('syn_large_20k', X, y,
          'AD', 'spherical', 'uniform', 'low', 'large', 'sklearn_gen', 7)

_try_uci('letter', 59,
         'C', 'mixed', 'uniform', 'low', 'large', 7)

_try_adbench('annthyroid',
             'AD', 'mixed', 'varying', 'low', 'large', 7)

_try_adbench('shuttle',
             'AD', 'mixed', 'uniform', 'low', 'large', 7)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

def print_summary():
    from collections import Counter

    cond_names = {
        1: 'Spherical (baseline)',
        2: 'Elongated / elliptical',
        3: 'Crescent / irregular',
        4: 'Nested / concentric',
        5: 'Varying density',
        6: 'High-dimensional',
        7: 'Large (efficiency)',
    }

    print(f'\n{"="*72}')
    print(f'  DATASET SUMMARY — {len(DATASETS)} loaded')
    print(f'{"="*72}')

    all_ok = True
    for cond, label in cond_names.items():
        dsets = [d for d in DATASETS.values() if d['condition'] == cond]
        n_c  = sum(1 for d in dsets if d['task'] == 'C')
        n_ad = sum(1 for d in dsets if d['task'] == 'AD')
        ok   = '✓' if len(dsets) >= 2 else '⚠ '
        if len(dsets) < 2:
            all_ok = False

        print(f'\n  {ok}  Cond {cond} — {label} '
              f'({len(dsets)} | C:{n_c} AD:{n_ad})')
        for d in sorted(dsets, key=lambda x: x['name']):
            tag = 'synth' if d['source'] == 'sklearn_gen' else 'real '
            print(f'      [{tag}] {d["name"]:30s} '
                  f'n={d["n"]:6d}  feat={d["features"]:4d}  '
                  f'{d["dim_level"]:5s}  {d["size_level"]}')

    print(f'\n{"="*72}')
    print(f'  Tasks:   {dict(Counter(d["task"]      for d in DATASETS.values()))}')
    print(f'  Shapes:  {dict(Counter(d["shape"]     for d in DATASETS.values()))}')
    print(f'  Dims:    {dict(Counter(d["dim_level"] for d in DATASETS.values()))}')
    print(f'  Sizes:   {dict(Counter(d["size_level"]for d in DATASETS.values()))}')
    print(f'  Sources: {dict(Counter(d["source"]    for d in DATASETS.values()))}')
    status = 'ALL CONDITIONS MET' if all_ok else 'DOWNLOAD ADBENCH FILES TO COMPLETE'
    print(f'\n  {status}')
    print(f'{"="*72}\n')


if __name__ == '__main__':
    print_summary()