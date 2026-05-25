"""
src/partitions.py  —  IK Partitioning Study
=============================================
All partitioning mechanisms x 2 kernel types (IK and IDK).
No C++ compiler or isotree required.

PARTITIONS (Cao et al. 2025, Table 1):
  anne       Voronoi (aNNE)             Section 3.2.3
  inne       Hypersphere (iNNE)         Section 3.2.2
  inne-overlapping  Hypersphere (iNNE) Overlapping
  iforest    Axis-parallel (iForest)    Section 3.1
  sciforest  Random hyperplane          Section 3.2.1

KERNEL TYPES:
  IK   κ(x,y)   = (1/t) <Φ(x), Φ(y)>           — point similarity
  IDK  K(Di,Dj) = (1/t) <KME(Di), KME(Dj)>     — distribution similarity

Usage:
    from src.partitions import get_partition

    part = get_partition('anne', n_estimators=200, max_samples=16)
    part.fit(X_train)

    K_ik   = part.similarity_ik(X_test)       # IK  matrix  (n × n)
    K_idk  = part.similarity_idk(X_test)      # IDK matrix  (n × n)
    sim    = part.idk_between(D1, D2)         # IDK scalar (group vs group)
    scores = part.idk_scores(X_test)          # anomaly scores (n,)
"""

import time
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from ikpykit.kernel._ik_anne import IK_ANNE

try:
    from ._ik_inne import IK_INNE
except ImportError:
    from _ik_inne import IK_INNE


# ── Metadata ───────────────────────────────────────────────────────────────

PARTITION_NAMES = {
    "anne": "Voronoi (aNNE)",
    "inne": "Hypersphere (iNNE)",
    "inne-overlapping": "Hypersphere (iNNE) Overlapping",
    "iforest": "Axis-parallel (iForest)",
    "sciforest": "Random hyperplane (SCiForest)",
}

PARTITION_GEOMETRY = {
    "anne": "Voronoi cells — nearest centroid assignment",
    "inne": "Hyperspheres — radius = NN distance of centroid",
    "iforest": "Hyper-rectangles — axis-aligned recursive splits",
    "sciforest": "Oblique partitions — random linear combination splits",
}

PARTITION_PAPERS = {
    "anne": "Qin et al. (AAAI 2019)",
    "inne": "Bandaragoda et al. (CIJ 2018)",
    "iforest": "Liu et al. (ICDM 2008)",
    "sciforest": "Liu et al. (ECML 2010)",
}


# ══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def _ik_sim(phi_X, phi_Y, n_est):
    K = (phi_X @ phi_Y.T) / n_est
    return K.toarray() if sparse.issparse(K) else np.array(K)


def _kme(phi):
    return np.asarray(phi.mean(axis=0)).ravel()


def _idk_scalar(kme_i, kme_j, n_est, normalize=True):
    raw = np.dot(kme_i, kme_j) / n_est
    if normalize:
        ni = np.sqrt(np.dot(kme_i, kme_i) / n_est)
        nj = np.sqrt(np.dot(kme_j, kme_j) / n_est)
        return raw / (ni * nj) if ni * nj > 0 else 0.0
    return raw


# ── Fixed-width leaf mapper (needed for iforest + sciforest) ──────────────

class _FixedLeafMapper:
    def fit(self, leaves_train):
        n, n_est = leaves_train.shape
        self._offsets = []
        self._uid_arrays = []   # sorted unique leaf IDs per tree
        self._col_arrays = []   # corresponding column indices per tree
        total = 0
        for t in range(n_est):
            uids = np.unique(leaves_train[:, t])          # already sorted by np.unique
            cols = np.arange(len(uids), dtype=np.int32)  # 0, 1, 2, ...
            self._uid_arrays.append(uids.astype(np.int64))
            self._col_arrays.append(cols)
            self._offsets.append(total)
            total += len(uids)
        self._total_cells = total
        self._n_est = n_est
        # _maps dict no longer needed — uid_arrays/col_arrays are used directly
        return self

    def transform(self, leaves_matrix):
        # FIX 🟡: vectorized — no Python loop over individual samples
        n_samples = leaves_matrix.shape[0]
        rows_list, cols_list = [], []
        for t in range(self._n_est):
            uids = self._uid_arrays[t]
            col_map = self._col_arrays[t]
            off = self._offsets[t]
            leaf_col = leaves_matrix[:, t].astype(np.int64)
            # searchsorted finds where each leaf_id sits in the sorted uid array
            pos = np.searchsorted(uids, leaf_col)
            # clip to valid range (unseen leaves at test time map to column 0)
            pos = np.clip(pos, 0, len(uids) - 1)
            matched = uids[pos] == leaf_col          # True where leaf was seen in training
            mapped_cols = np.where(matched, col_map[pos] + off, off)  # unseen → col 0
            rows_list.append(np.arange(n_samples))
            cols_list.append(mapped_cols)
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(n_samples, self._total_cells)
        )


# ══════════════════════════════════════════════════════════════════════════
# BASE PARTITION CLASS
# ══════════════════════════════════════════════════════════════════════════

class _BasePartition(TransformerMixin, BaseEstimator):
    def __init__(self, n_estimators=200, max_samples=16, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X).astype(np.float32)
        self._fit_partition(X)
        phi_train = self._transform_partition(X)
        self._train_kme = _kme(phi_train)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")
        return self._transform_partition(check_array(X).astype(np.float32))

    # ── IK ─────────────────────────────────────────────────────────────────

    def similarity_ik(self, X, Y=None):
        phi_X = self.transform(X)
        phi_Y = self.transform(Y) if Y is not None else phi_X
        return _ik_sim(phi_X, phi_Y, self.n_estimators)

    # ── IDK ────────────────────────────────────────────────────────────────

    def similarity_idk(self, X, normalize=True):
        phi_X = self.transform(X)
        K = _ik_sim(phi_X, phi_X, self.n_estimators)
        if normalize:
            d = np.sqrt(np.diag(K))
            d[d == 0] = 1.0
            K = K / np.outer(d, d)
        return np.clip(K, 0, 1)

    def idk_between(self, Di, Dj, normalize=True):
        phi_i = self.transform(Di)
        phi_j = self.transform(Dj)
        return _idk_scalar(_kme(phi_i), _kme(phi_j), self.n_estimators, normalize)

    def idk_scores(self, X, normalize=True):
        check_is_fitted(self, "is_fitted_")
        phi_X = self.transform(X)                          # (n, d) sparse
        g_kme = self._train_kme                            # (d,) from fit()

        # Vectorized IDK: each row of phi_X dotted with g_kme
        # raw(i) = phi_X[i] · g_kme / n_est
        phi_dense = np.asarray(phi_X.dot(g_kme)).ravel()  # (n,)  sparse × dense
        raw = phi_dense / self.n_estimators

        if normalize:
            # norm of each point's KME = sqrt(phi_X[i]·phi_X[i] / n_est)
            phi_sq = np.asarray(phi_X.power(2).sum(axis=1)).ravel()  # (n,)
            ni = np.sqrt(phi_sq / self.n_estimators)
            # norm of training KME
            nj = np.sqrt(np.dot(g_kme, g_kme) / self.n_estimators)
            denom = ni * nj
            similarity = np.where(denom > 0, raw / denom, 0.0)
        else:
            similarity = raw

        return np.clip(1.0 - similarity, 0, 1)

    def similarity(self, X, Y=None, kernel="ik"):
        """Convenience wrapper. kernel='ik' or 'idk'."""
        if kernel == "ik":
            return self.similarity_ik(X, Y)
        return self.similarity_idk(X)

    def _fit_partition(self, X):
        raise NotImplementedError

    def _transform_partition(self, X):
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════
# PARTITION 1 — VORONOI (aNNE)
# ══════════════════════════════════════════════════════════════════════════

class VoronoiPartition(_BasePartition):
    """Voronoi (aNNE) — ikpykit IK_ANNE — Section 3.2.3."""

    def _fit_partition(self, X):
        self._model = IK_ANNE(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        ).fit(X)

    def _transform_partition(self, X):
        return self._model.transform(X)


# ══════════════════════════════════════════════════════════════════════════
# PARTITION 2 — HYPERSPHERE (iNNE)
# ══════════════════════════════════════════════════════════════════════════

class HyperspherePartition(_BasePartition):
    """Hypersphere (iNNE) — ikpykit IK_INNE — Section 3.2.2."""

    def _fit_partition(self, X):
        self._model = IK_INNE(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        ).fit(X)

    def _transform_partition(self, X):
        return self._model.transform(X)


class HyperspherePartitionOverlapping(_BasePartition):
    """Hypersphere (iNNE) with overlapping regions — Section 3.2.2."""

    def _fit_partition(self, X):
        self._model = IK_INNE(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            overlapping=True,
        ).fit(X)

    def _transform_partition(self, X):
        return self._model.transform(X)


# ══════════════════════════════════════════════════════════════════════════
# PARTITION 3 — AXIS-PARALLEL (iForest)
# ══════════════════════════════════════════════════════════════════════════

class AxisParallelPartition(_BasePartition):
    """Axis-parallel (iForest) — sklearn IsolationForest — Section 3.1."""

    def _fit_partition(self, X):
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, X.shape[0]),
            random_state=self.random_state,
        ).fit(X)
        # Learn fixed leaf mapping from training data
        leaves_tr = np.column_stack(
            [t.apply(X, check_input=False) for t in self._model.estimators_]
        )
        self._mapper = _FixedLeafMapper().fit(leaves_tr)

    def _transform_partition(self, X):
        leaves = np.column_stack(
            [t.apply(X, check_input=False) for t in self._model.estimators_]
        )
        return self._mapper.transform(leaves)


# ══════════════════════════════════════════════════════════════════════════
# PARTITION 4 — RANDOM HYPERPLANE (SCiForest) — pure numpy
# ══════════════════════════════════════════════════════════════════════════

class _SCiTree:
    """Single oblique-split tree (Liu et al. ECML 2010)."""

    def __init__(self, max_depth, n_dims, rng):
        self.max_depth = max_depth
        self.n_dims = n_dims
        self.rng = rng

    def _build(self, X, depth=0):
        n, d = X.shape
        if depth >= self.max_depth or n <= 1:
            return {"leaf": True, "id": None}
        feat = self.rng.choice(d, min(self.n_dims, d), replace=False)
        coef = self.rng.randn(len(feat))
        proj = X[:, feat] @ coef
        lo, hi = proj.min(), proj.max()
        if lo >= hi:
            return {"leaf": True, "id": None}
        split = self.rng.uniform(lo, hi)
        mask = proj <= split
        return {
            "leaf": False,
            "feat": feat,
            "coef": coef,
            "split": split,
            "left": self._build(X[mask], depth + 1),
            "right": self._build(X[~mask], depth + 1),
        }

    def _ids(self, node, cid=0):
        if node["leaf"]:
            node["id"] = cid
            return cid + 1
        cid = self._ids(node["left"], cid)
        cid = self._ids(node["right"], cid)
        return cid

    def fit(self, X):
        self._tree = self._build(X)
        self._ids(self._tree)
        self._nodes_feat  = []   # list of feat arrays, one per internal node
        self._nodes_coef  = []
        self._nodes_split = []
        self._nodes_left  = []   # child node index (-1 = leaf)
        self._nodes_right = []
        self._nodes_leaf_id = [] # leaf id (-1 = internal node)
        self._compile(self._tree)
        self._nodes_split = np.array(self._nodes_split, dtype=np.float32)
        self._nodes_left  = np.array(self._nodes_left,  dtype=np.int32)
        self._nodes_right = np.array(self._nodes_right, dtype=np.int32)
        self._nodes_leaf_id = np.array(self._nodes_leaf_id, dtype=np.int32)
        return self

    def _compile(self, node):
        if node["leaf"]:
            self._nodes_feat.append(None)
            self._nodes_coef.append(None)
            self._nodes_split.append(0.0)
            self._nodes_left.append(-1)
            self._nodes_right.append(-1)
            self._nodes_leaf_id.append(node["id"])
        else:
            idx = len(self._nodes_split)
            # Reserve slot first, fill children recursively
            self._nodes_feat.append(node["feat"])
            self._nodes_coef.append(node["coef"])
            self._nodes_split.append(node["split"])
            self._nodes_left.append(-1)   # placeholder
            self._nodes_right.append(-1)  # placeholder
            self._nodes_leaf_id.append(-1)
            left_idx = len(self._nodes_split)
            self._compile(node["left"])
            right_idx = len(self._nodes_split)
            self._compile(node["right"])
            self._nodes_left[idx]  = left_idx
            self._nodes_right[idx] = right_idx

    def apply(self, X):
        n = len(X)
        node_ids = np.zeros(n, dtype=np.int32)   # all start at root (node 0)
        leaf_ids  = np.full(n, -1, dtype=np.int32)
        active    = np.ones(n, dtype=bool)

        while active.any():
            active_idx = np.where(active)[0]
            cur        = node_ids[active_idx]

            # ── settle samples that landed on a leaf ──────────────────────
            is_leaf              = self._nodes_leaf_id[cur] >= 0
            done_idx             = active_idx[is_leaf]
            leaf_ids[done_idx]   = self._nodes_leaf_id[cur[is_leaf]]
            active[done_idx]     = False

            # ── route samples still in internal nodes ─────────────────────
            live_mask            = ~is_leaf
            live_sample_idx      = active_idx[live_mask]   # global sample indices
            live_node_ids        = cur[live_mask]           # which node each is at

            if len(live_sample_idx) == 0:
                break

            go_left = np.zeros(len(live_sample_idx), dtype=bool)

            # Group samples by node so each node gets ONE matrix multiply.
            # unique_nodes: the distinct internal nodes that have ≥1 sample.
            unique_nodes = np.unique(live_node_ids)
            for ni in unique_nodes:
                mask   = live_node_ids == ni           # samples at this node
                feat   = self._nodes_feat[ni]          # feature indices (variable length)
                coef   = self._nodes_coef[ni]          # coefficients
                # X[live_sample_idx[mask]][:, feat] is (k, len(feat))
                # @ coef is (k,) — one matrix multiply for all k samples at ni
                proj   = X[live_sample_idx[mask]][:, feat] @ coef
                go_left[mask] = proj <= self._nodes_split[ni]

            node_ids[live_sample_idx] = np.where(
                go_left,
                self._nodes_left[live_node_ids],
                self._nodes_right[live_node_ids],
            )

        return leaf_ids


class RandomHyperplanePartition(_BasePartition):
    """Random hyperplane (SCiForest) — pure numpy — Section 3.2.1."""

    def __init__(
        self, n_estimators=200, max_samples=16, random_state=42, n_dims=2, max_depth=8
    ):
        super().__init__(n_estimators, max_samples, random_state)
        self.n_dims = n_dims
        self.max_depth = max_depth

    def _fit_partition(self, X):
        rng = check_random_state(self.random_state)
        n = X.shape[0]
        subs = min(self.max_samples, n)
        self._trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, subs, replace=False)
            self._trees.append(_SCiTree(self.max_depth, self.n_dims, rng).fit(X[idx]))
        # Learn fixed leaf mapping from training data
        leaves_tr = np.column_stack([t.apply(X) for t in self._trees])
        self._mapper = _FixedLeafMapper().fit(leaves_tr)

    def _transform_partition(self, X):
        leaves = np.column_stack([t.apply(X) for t in self._trees])
        return self._mapper.transform(leaves)


# ══════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════

_CLASSES = {
    "anne": VoronoiPartition,
    "inne": HyperspherePartition,
    "inne-overlapping": HyperspherePartitionOverlapping,
    "iforest": AxisParallelPartition,
    "sciforest": RandomHyperplanePartition,
}


def get_partition(
    method, kernel="ik", n_estimators=200, max_samples=16, random_state=42, **kwargs
):
    """
    Return an unfitted partition object.

    Parameters
    ----------
    method       : 'anne' | 'inne' | 'inne-overlapping' | 'iforest' | 'sciforest'
    kernel       : 'ik' | 'idk'  (informational — both always available)
    n_estimators : t  in the paper (default 200)
    max_samples  : psi  in the paper (default 16)
    random_state : seed (default 42)
    **kwargs     : e.g. n_dims=3 for sciforest

    After part.fit(X_train):
        part.similarity_ik(X_test)       → IK  matrix (n×n)
        part.similarity_idk(X_test)      → IDK matrix (n×n)
        part.idk_between(Di, Dj)         → IDK scalar (group vs group)
        part.idk_scores(X_test)          → anomaly scores (n,)
    """
    method = method.lower()
    if method not in _CLASSES:
        raise ValueError(f"Unknown '{method}'. Valid: {list(_CLASSES)}")
    part = _CLASSES[method](
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=random_state,
        **kwargs,
    )
    part._kernel_type = kernel
    return part


# ══════════════════════════════════════════════════════════════════════════
# SANITY CHECK — python src/partitions.py
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 66)
    print("  IK Partitioning Study — src/partitions.py sanity check")
    print("=" * 66)
    np.random.seed(42)
    X_tr = np.random.rand(300, 10).astype(np.float32)
    X_te = np.random.rand(100, 10).astype(np.float32)
    D1, D2 = X_te[:50], X_te[50:]
    N_EST, N_SUB = 50, 16
    all_ok = True

    for method in ["anne", "inne", "inne-overlapping", "iforest", "sciforest"]:
        try:
            part = get_partition(
                method, n_estimators=N_EST, max_samples=N_SUB, random_state=42
            )
            t0 = time.perf_counter()
            part.fit(X_tr)
            fit_t = time.perf_counter() - t0

            K_ik = part.similarity_ik(X_te)
            K_idk = part.similarity_idk(X_te)
            idk_s = part.idk_between(D1, D2)
            scores = part.idk_scores(X_te)

            ok = (
                K_ik.min() >= 0
                and K_ik.max() <= 1.001
                and K_idk.min() >= 0
                and K_idk.max() <= 1.001
                and 0 <= idk_s <= 1.001
                and scores.min() >= 0
                and scores.max() <= 1.001
            )
            if not ok:
                all_ok = False

            print(f"\n  {PARTITION_NAMES[method]}")
            print(f"    paper       : {PARTITION_PAPERS.get(method, 'N/A')}")
            print(f"    fit time    : {fit_t:.4f}s")
            print(f"    IK  K range : [{K_ik.min():.3f}, {K_ik.max():.3f}]")
            print(f"    IDK K range : [{K_idk.min():.3f}, {K_idk.max():.3f}]")
            print(f"    IDK(D1,D2)  : {idk_s:.4f}  (group similarity)")
            print(f"    IDK scores  : [{scores.min():.3f}, {scores.max():.3f}]")
            print(f"    status      : {'OK ✓' if ok else 'FAIL ✗'}")

        except Exception as e:
            import traceback

            print(f"\n  {method}: FAILED")
            traceback.print_exc()
            all_ok = False

    print()
    print("=" * 66)
    print(f"  {'ALL 5 × IK + IDK OK ✓' if all_ok else 'SOME FAILED — see above'}")
    print()
    print("  Usage:")
    print("    from src.partitions import get_partition")
    print("    part = get_partition('anne', n_estimators=200, max_samples=16)")
    print("    part.fit(X_train)")
    print("    K_ik   = part.similarity_ik(X_test)    # IK  kernel matrix")
    print("    K_idk  = part.similarity_idk(X_test)   # IDK kernel matrix")
    print("    sim    = part.idk_between(D1, D2)       # group similarity")
    print("    scores = part.idk_scores(X_test)        # anomaly scores")
    print("=" * 66)