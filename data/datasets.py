"""
datasets.py  —  IK Partitioning Study
======================================
Registers all datasets for the study. Data is loaded lazily on first access.

Sources
-------
- sklearn builtins/generators : always available, no download
- ucimlrepo                   : pip install ucimlrepo  (needs internet)
- ADBench (ODDS datasets)     : auto-downloaded from GitHub on first run
                                saved to data/anomaly_detection/

Usage
-----
    from data.datasets import DATASETS, print_summary
    print_summary()          # shows metadata without loading any data

    ds = DATASETS['iris']    # triggers load of 'iris' only
    X, y = ds['X'], ds['y']  # normalized X, integer y
    task = ds['task']        # 'C' or 'AD'
    cond = ds['condition']   # 1-7

    for name, ds in DATASETS.items():   # each dataset loaded as iterated
        X, y = ds['X'], ds['y']
"""

import os
import io
import ssl
import socket
import warnings
import urllib.request

# UCI ML Repo SSL certificate is sometimes expired;
# patch urllib so ucimlrepo (and pandas) bypass SSL verification.
_orig_urlopen = urllib.request.urlopen

def _patched_urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, **kwargs):
    kwargs['context'] = ssl._create_unverified_context()
    return _orig_urlopen(url, data, timeout, **kwargs)

urllib.request.urlopen = _patched_urlopen

import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_blobs, make_moons, make_circles,
    make_gaussian_quantiles, make_classification,
)

warnings.filterwarnings('ignore')

# ── ADBench filename map ───────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
# LAZY DATASET CONTAINER
# ══════════════════════════════════════════════════════════════════════════

class _LazyDataset:
    """
    Holds dataset metadata immediately; defers loading X/y until first access.

    Behaves like a plain dict: ds['X'], ds['y'], ds['task'], ds['condition'], ...
    """

    def __init__(self, name, loader, task, shape, density,
                 dim_level, size_level, source, condition):
        self._loader = loader
        self._data   = None           # populated on first access
        self._meta = dict(
            name=name, task=task,
            shape=shape, density=density,
            dim_level=dim_level, size_level=size_level,
            source=source, condition=condition,
        )

    def _ensure_loaded(self):
        if self._data is None:
            self._load()

    def _load(self):
        result = self._loader()
        if result is None:
            raise RuntimeError(
                f"Dataset '{self._meta['name']}' could not be loaded.")
        X_raw, y_raw = result
        X = MinMaxScaler().fit_transform(np.nan_to_num(X_raw.astype(float)))
        y = LabelEncoder().fit_transform(np.array(y_raw).ravel())
        name = self._meta['name']
        src_tag = '[synth]' if self._meta['source'] == 'sklearn_gen' else '[real ]'
        print(f'  OK   {src_tag} {name:28s} '
              f'n={X.shape[0]:6d}  feat={X.shape[1]:4d}  '
              f'task={self._meta["task"]}  cond={self._meta["condition"]}')
        self._data = dict(X=X, y=y, n=X.shape[0], features=X.shape[1])

    def __getitem__(self, key):
        if key in ('X', 'y', 'n', 'features'):
            self._ensure_loaded()
            return self._data[key]
        return self._meta[key]

    def __contains__(self, key):
        return key in self._meta or key in ('X', 'y', 'n', 'features')

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        self._ensure_loaded()
        return {**self._meta, **self._data}.keys()

    def values(self):
        self._ensure_loaded()
        return {**self._meta, **self._data}.values()

    def items(self):
        self._ensure_loaded()
        return {**self._meta, **self._data}.items()

    def __repr__(self):
        state = '(loaded)' if self._data is not None else '(not loaded)'
        return f"<LazyDataset '{self._meta['name']}' cond={self._meta['condition']} {state}>"


class _LazyDatasetDict(dict):
    """dict subclass; yields _LazyDataset values without auto-loading."""

    def items(self):
        for k, v in super().items():
            yield k, v

    def loaded(self):
        """Return mapping of already-loaded datasets (no side effects)."""
        return {k: v for k, v in super().items() if v._data is not None}


DATASETS = _LazyDatasetDict()


# ══════════════════════════════════════════════════════════════════════════
# REGISTRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _register(name, loader, task, shape, density,
              dim_level, size_level, source, condition):
    """Register a dataset with a loader callable. No data is loaded yet."""
    DATASETS[name] = _LazyDataset(
        name=name, loader=loader, task=task,
        shape=shape, density=density,
        dim_level=dim_level, size_level=size_level,
        source=source, condition=condition,
    )


def _download_adbench(name):
    fname = ADBENCH_FILES.get(name)
    if fname is None:
        return None
    os.makedirs(SAVE_DIR, exist_ok=True)
    dest = os.path.join(SAVE_DIR, fname)
    if os.path.exists(dest):
        return dest
    url = f'{ADBENCH_BASE}/{fname}'
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req, context=ctx, timeout=20).read()
        np.load(io.BytesIO(data), allow_pickle=True)
        with open(dest, 'wb') as f:
            f.write(data)
        return dest
    except Exception:
        return None


def _load_adbench(name):
    fname = ADBENCH_FILES.get(name)
    if fname is None:
        return None
    local = os.path.join(SAVE_DIR, fname)
    if not os.path.exists(local):
        print(f'         downloading {name}...', end=' ', flush=True)
        local = _download_adbench(name)
        print('OK' if local else 'FAILED')
    if local and os.path.exists(local):
        d = np.load(local, allow_pickle=True)
        return d['X'].astype(float), d['y'].ravel().astype(int)
    return None


def _load_uci(dataset_id):
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=dataset_id)
    X = ds.data.features.values.astype(float)
    y = ds.data.targets.values.ravel()
    return X, y


def _register_uci(name, dataset_id, task, shape, density,
                  dim_level, size_level, condition):
    def _loader():
        try:
            return _load_uci(dataset_id)
        except Exception as e:
            print(f'  SKIP {name:30s} (ucimlrepo id={dataset_id}): {str(e)[:60]}')
            return None
    _register(name, _loader, task, shape, density,
              dim_level, size_level, 'UCI', condition)


def _register_adbench(name, task, shape, density,
                      dim_level, size_level, condition):
    def _loader():
        result = _load_adbench(name)
        if result is None:
            print(f'  SKIP {name:30s} (place {ADBENCH_FILES.get(name,"")} in {SAVE_DIR}/)')
        return result
    _register(name, _loader, task, shape, density,
              dim_level, size_level, 'ADBench', condition)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 1 — Spherical clusters
# ══════════════════════════════════════════════════════════════════════════

_register('iris',
          lambda: (load_iris().data, load_iris().target),
          'C', 'spherical', 'uniform', 'low', 'small', 'sklearn', 1)

_register('breast_cancer',
          lambda: (load_breast_cancer().data, load_breast_cancer().target),
          'AD', 'spherical', 'uniform', 'mid', 'medium', 'sklearn', 1)

_register('syn_blobs_small',
          lambda: make_blobs(n_samples=500, centers=3, cluster_std=0.8, random_state=42),
          'C', 'spherical', 'uniform', 'low', 'small', 'sklearn_gen', 1)

_register('syn_blobs_medium',
          lambda: make_blobs(n_samples=3000, centers=4, cluster_std=1.0, random_state=42),
          'C', 'spherical', 'uniform', 'low', 'medium', 'sklearn_gen', 1)

_register_uci('wbc_uci', 15, 'AD', 'spherical', 'uniform', 'mid', 'small', 1)
_register_adbench('breastw', 'AD', 'spherical', 'uniform', 'low', 'small', 1)
_register_adbench('wbc', 'AD', 'spherical', 'uniform', 'mid', 'small', 1)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 2 — Elongated / elliptical clusters
# ══════════════════════════════════════════════════════════════════════════

_register('wine',
          lambda: (load_wine().data, load_wine().target),
          'C', 'elliptical', 'uniform', 'low', 'small', 'sklearn', 2)

_register('syn_elongated_small',
          lambda: make_blobs(n_samples=500, centers=3,
                             cluster_std=[3.0, 0.5, 2.0], random_state=42),
          'C', 'elliptical', 'uniform', 'low', 'small', 'sklearn_gen', 2)

_register('syn_elongated_medium',
          lambda: make_blobs(n_samples=800, centers=4,
                             cluster_std=[4.0, 0.3, 2.5, 1.0], random_state=42),
          'C', 'elliptical', 'uniform', 'low', 'medium', 'sklearn_gen', 2)

_register_uci('vehicle', 149, 'C', 'elliptical', 'uniform', 'mid', 'medium', 2)
_register_uci('glass', 42, 'C', 'elliptical', 'uniform', 'low', 'small', 2)
_register_adbench('ionosphere', 'AD', 'elliptical', 'uniform', 'mid', 'small', 2)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 3 — Crescent / chain / irregular
# ══════════════════════════════════════════════════════════════════════════

_register('syn_moons_small',
          lambda: make_moons(n_samples=500, noise=0.05, random_state=42),
          'C', 'crescent', 'uniform', 'low', 'small', 'sklearn_gen', 3)

_register('syn_moons_medium',
          lambda: make_moons(n_samples=1000, noise=0.10, random_state=42),
          'C', 'crescent', 'uniform', 'low', 'medium', 'sklearn_gen', 3)


def _make_moons_ad():
    X_in, _ = make_moons(n_samples=950, noise=0.05, random_state=42)
    rng = np.random.RandomState(0)
    X_out = rng.uniform(-2, 3, (50, 2))
    return np.vstack([X_in, X_out]), np.array([0]*950 + [1]*50)


_register('syn_moons_ad', _make_moons_ad,
          'AD', 'crescent', 'uniform', 'low', 'medium', 'sklearn_gen', 3)

_register_adbench('vowels', 'AD', 'irregular', 'uniform', 'low', 'medium', 3)
_register_adbench('lympho', 'AD', 'irregular', 'sparse', 'low', 'small', 3)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 4 — Nested / concentric
# ══════════════════════════════════════════════════════════════════════════

_register('syn_circles_small',
          lambda: make_circles(n_samples=500, noise=0.05, factor=0.4, random_state=42),
          'C', 'nested', 'uniform', 'low', 'small', 'sklearn_gen', 4)

_register('syn_circles_medium',
          lambda: make_circles(n_samples=800, noise=0.08, factor=0.5, random_state=42),
          'C', 'nested', 'uniform', 'low', 'medium', 'sklearn_gen', 4)

_register('syn_gauss_quantiles',
          lambda: make_gaussian_quantiles(n_samples=500, n_features=2,
                                         n_classes=3, random_state=42),
          'C', 'nested', 'uniform', 'low', 'small', 'sklearn_gen', 4)


def _make_circles_ad():
    """Concentric circles: points on the rings = normal, random background = anomaly."""
    X_in, _ = make_circles(n_samples=950, noise=0.05, factor=0.4, random_state=42)
    rng = np.random.RandomState(0)
    X_out = rng.uniform(-1.5, 1.5, (50, 2))
    return np.vstack([X_in, X_out]), np.array([0]*950 + [1]*50)

_register('syn_circles_ad', _make_circles_ad,
          'AD', 'nested', 'uniform', 'low', 'small', 'sklearn_gen', 4)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 5 — Varying density
# ══════════════════════════════════════════════════════════════════════════

def _make_density_2d():
    X1, _ = make_blobs(400, centers=[[0, 0]], cluster_std=0.3, random_state=42)
    X2, _ = make_blobs(100, centers=[[6, 6]], cluster_std=2.5, random_state=42)
    return np.vstack([X1, X2]), np.array([0]*400 + [1]*100)


def _make_density_3d():
    X1, _ = make_blobs(300, centers=[[0, 0, 0]], cluster_std=0.2, random_state=42)
    X2, _ = make_blobs(300, centers=[[5, 5, 5]], cluster_std=1.5, random_state=42)
    X3, _ = make_blobs(400, centers=[[10, 0, 5]], cluster_std=3.0, random_state=42)
    return np.vstack([X1, X2, X3]), np.array([0]*300 + [1]*300 + [2]*400)


_register('syn_density_2d', _make_density_2d,
          'C', 'mixed', 'varying', 'low', 'small', 'sklearn_gen', 5)

_register('syn_density_3d', _make_density_3d,
          'C', 'mixed', 'varying', 'low', 'medium', 'sklearn_gen', 5)

_register_uci('ecoli', 39, 'C', 'mixed', 'varying', 'low', 'small', 5)
_register_uci('yeast', 110, 'C', 'mixed', 'varying', 'low', 'medium', 5)
_register_adbench('thyroid', 'AD', 'mixed', 'varying', 'low', 'medium', 5)
_register_adbench('cardio', 'AD', 'mixed', 'varying', 'mid', 'medium', 5)
_register_adbench('waveform', 'C', 'mixed', 'varying', 'mid', 'medium', 5)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 6 — High dimensionality
# ══════════════════════════════════════════════════════════════════════════

_register('digits',
          lambda: (load_digits().data, load_digits().target),
          'C', 'mixed', 'uniform', 'high', 'large', 'sklearn', 6)

_register('syn_highdim_50',
          lambda: make_classification(n_samples=1000, n_features=50,
              n_informative=20, n_redundant=10,
              n_classes=3, n_clusters_per_class=1, random_state=42),
          'C', 'spherical', 'uniform', 'high', 'medium', 'sklearn_gen', 6)

_register('syn_highdim_100',
          lambda: make_classification(n_samples=1000, n_features=100,
              n_informative=30, n_redundant=20,
              n_classes=4, n_clusters_per_class=1, random_state=42),
          'C', 'spherical', 'uniform', 'vhigh', 'medium', 'sklearn_gen', 6)

_register_uci('dermatology', 33, 'C', 'mixed', 'uniform', 'high', 'small', 6)
_register_adbench('satellite', 'AD', 'mixed', 'varying', 'high', 'large', 6)
_register_adbench('musk', 'AD', 'mixed', 'uniform', 'vhigh', 'medium', 6)
_register_adbench('optdigits', 'AD', 'mixed', 'uniform', 'high', 'medium', 6)
_register_adbench('pendigits', 'AD', 'mixed', 'uniform', 'low', 'large', 6)


# ══════════════════════════════════════════════════════════════════════════
# CONDITION 7 — Large datasets (computational efficiency)
# ══════════════════════════════════════════════════════════════════════════

def _make_large_10k():
    X, y = make_blobs(n_samples=10000, centers=5, cluster_std=1.5, random_state=42)
    return X, (y == 0).astype(int)


def _make_large_20k():
    X, y = make_blobs(n_samples=20000, centers=5, cluster_std=1.5, random_state=42)
    return X, (y == 0).astype(int)


_register('syn_large_10k', _make_large_10k,
          'AD', 'spherical', 'uniform', 'low', 'large', 'sklearn_gen', 7)

_register('syn_large_20k', _make_large_20k,
          'AD', 'spherical', 'uniform', 'low', 'large', 'sklearn_gen', 7)

_register_uci('letter', 59, 'C', 'mixed', 'uniform', 'low', 'large', 7)
_register_adbench('annthyroid', 'AD', 'mixed', 'varying', 'low', 'large', 7)
_register_adbench('shuttle', 'AD', 'mixed', 'uniform', 'low', 'large', 7)


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY  (metadata only — no data is loaded)
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
    print(f'  DATASET REGISTRY  — {len(DATASETS)} registered  '
          f'({len(DATASETS.loaded())} loaded)')
    print(f'{"="*72}')

    all_ok = True
    for cond, label in cond_names.items():
        dsets = [v for v in DATASETS.values() if v['condition'] == cond]
        n_c   = sum(1 for d in dsets if d['task'] == 'C')
        n_ad  = sum(1 for d in dsets if d['task'] == 'AD')
        ok    = chr(10003) if len(dsets) >= 2 else chr(9888)
        if len(dsets) < 2:
            all_ok = False

        print(f'\n  {ok}  Cond {cond} — {label} ({len(dsets)} | C:{n_c} AD:{n_ad})')
        for d in sorted(dsets, key=lambda x: x['name']):
            marker = '*' if d._data is not None else ' '
            tag = 'synth' if d['source'] == 'sklearn_gen' else 'real '
            if d._data is not None:
                print(f'    {marker} [{tag}] {d["name"]:30s} '
                      f'n={d["n"]:6d}  feat={d["features"]:4d}  '
                      f'{d["dim_level"]:5s}  {d["size_level"]}')
            else:
                print(f'    {marker} [{tag}] {d["name"]:30s} '
                      f'(not loaded)  {d["dim_level"]:5s}  {d["size_level"]}')

    print(f'\n{"="*72}')
    print(f'  Tasks:   {dict(Counter(d["task"]       for d in DATASETS.values()))}')
    print(f'  Shapes:  {dict(Counter(d["shape"]      for d in DATASETS.values()))}')
    print(f'  Dims:    {dict(Counter(d["dim_level"]  for d in DATASETS.values()))}')
    print(f'  Sizes:   {dict(Counter(d["size_level"] for d in DATASETS.values()))}')
    print(f'  Sources: {dict(Counter(d["source"]     for d in DATASETS.values()))}')
    status = 'ALL CONDITIONS MET' if all_ok else 'DOWNLOAD ADBENCH FILES TO COMPLETE'
    print(f'\n  {status}')
    print(f'{"="*72}\n')


if __name__ == '__main__':
    print_summary()
