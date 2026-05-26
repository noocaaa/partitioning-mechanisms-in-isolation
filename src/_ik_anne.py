import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IK_ANNE(TransformerMixin, BaseEstimator):
    """Build aNNE feature vectors via Voronoi-style weighted embeddings."""

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        random_state=None,
        k=1,
        weighting="boundary",
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.k = k
        self.weighting = weighting

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples = X.shape[0]
        self.max_samples_ = min(self.max_samples, n_samples)
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        self.center_ids = np.empty(
            (self.n_estimators, self.max_samples_), dtype=np.int32
        )
        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            self.center_ids[i] = rnd.choice(n_samples, self.max_samples_, replace=False)

        self.unique_ids = np.unique(self.center_ids)
        self.center_data = X[self.unique_ids]

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        weighting = str(self.weighting).lower()
        if weighting not in {"binary", "boundary"}:
            raise ValueError("weighting must be one of {'binary', 'boundary'}")

        n_samples = X.shape[0]
        n_features = self.n_estimators * self.max_samples_

        X_dists = euclidean_distances(X, self.center_data)
        id_to_index = {id_val: idx for idx, id_val in enumerate(self.unique_ids)}

        blocks = []
        for est_idx in range(self.n_estimators):
            centers = self.center_ids[est_idx]
            center_indices = np.array([id_to_index[center] for center in centers])
            estimator_dists = X_dists[:, center_indices]

            k = int(self.k)
            if k <= 0:
                raise ValueError("k must be a positive integer")
            k = min(k, estimator_dists.shape[1])

            if k == 1:
                nn_indices = np.argmin(estimator_dists, axis=1)
                rows = np.arange(n_samples)
                cols = nn_indices + (est_idx * self.max_samples_)
                data = np.ones(n_samples, dtype=np.float64)
            else:
                # Keep one extra neighbor when possible so boundary weighting can
                # reference an external boundary distance instead of the k-th
                # selected neighbor (which would force the last selected weight to 0).
                k_plus = min(k + 1, estimator_dists.shape[1])

                nn_plus_indices = np.argpartition(
                    estimator_dists, kth=k_plus - 1, axis=1
                )[:, :k_plus]
                nn_plus_dists = np.take_along_axis(
                    estimator_dists, nn_plus_indices, axis=1
                )
                order_plus = np.argsort(nn_plus_dists, axis=1)
                nn_plus_indices = np.take_along_axis(
                    nn_plus_indices, order_plus, axis=1
                )
                nn_plus_dists = np.take_along_axis(nn_plus_dists, order_plus, axis=1)

                nn_indices = nn_plus_indices[:, :k]
                nn_dists = nn_plus_dists[:, :k]
                order = np.argsort(nn_dists, axis=1)
                nn_indices = np.take_along_axis(nn_indices, order, axis=1)
                nn_dists = np.take_along_axis(nn_dists, order, axis=1)

                if weighting == "binary":
                    weights = np.ones_like(nn_dists)
                else:
                    # Boundary-aware weights relative to an external boundary distance.
                    # Prefer the (k+1)-th neighbor distance when available so the
                    # furthest selected centroid does not get forced to zero weight.
                    if k_plus > k:
                        boundary_dist = np.maximum(nn_plus_dists[:, [k]], MIN_FLOAT)
                    else:
                        boundary_dist = np.maximum(nn_dists[:, [-1]], MIN_FLOAT)
                    weights = np.maximum(1.0 - (nn_dists / boundary_dist), 0.0)

                    # Degenerate rows can happen when selected distances are all identical.
                    zero_rows = np.linalg.norm(weights, axis=1) == 0
                    if np.any(zero_rows):
                        weights[zero_rows] = 1.0

                row_norm = np.linalg.norm(weights, axis=1, keepdims=True)
                weights = np.divide(
                    weights,
                    row_norm,
                    out=np.zeros_like(weights),
                    where=row_norm > 0,
                )

                rows = np.repeat(np.arange(n_samples), k)
                cols = nn_indices.reshape(-1) + (est_idx * self.max_samples_)
                data = weights.reshape(-1)

            blocks.append(
                sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))
            )

        return sparse.hstack(blocks, format="csr")
