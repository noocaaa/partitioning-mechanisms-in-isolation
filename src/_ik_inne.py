import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.metrics._pairwise_distances_reduction import ArgKmin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps


class IK_INNE(TransformerMixin, BaseEstimator):
    """Build Isolation Kernel feature vector representations via the feature map
    for a given dataset.

    Isolation kernel is a data dependent kernel measure that is
    adaptive to local data distribution and has more flexibility in capturing
    the characteristics of the local data distribution. It has been shown promising
    performance on density and distance-based classification and clustering problems.

    This version uses iforest to split the data space and calculate Isolation
    kernel Similarity. Based on this implementation, the feature
    in the Isolation kernel space is the index of the cell in Voronoi diagrams. Each
    point is represented as a binary vector such that only the cell the point falling
    into is 1.

    Parameters
    ----------

    n_estimators : int
        The number of base estimators in the ensemble.

    max_samples : int
        The number of samples to draw from X to train each base estimator.

    random_state : int, RandomState instance or None, default=None
        Controls the pseudo-randomness of the selection of the feature
        and split values for each branching step and each tree in the forest.

    overlapping : bool, default=False
        If True, a point can belong to multiple hyperspheres (all that contain it).
        If False, each point is assigned only to its nearest centroid's hypersphere.

    References
    ----------
    .. [1] Qin, X., Ting, K.M., Zhu, Y. and Lee, V.C.
    "Nearest-neighbour-induced isolation similarity and its impact on density-based clustering".
    In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 33, 2019, July, pp. 4755-4762
    """

    def __init__(
        self,
        n_estimators,
        max_samples,
        random_state=None,
        overlapping=False,
        r=1.0,
        weighting="boundary",
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.overlapping = overlapping
        self.max_samples_ = None
        self._seeds = None
        self._radius = None
        self._centroids = None
        self.inclusive = overlapping
        self.r = r
        self.weighting = weighting

    def fit(self, X, y=None):
        X = check_array(X)
        self.max_samples_ = self.max_samples
        n_samples, n_features = X.shape
        self.max_samples_ = min(self.max_samples_, n_samples)

        self._centroids = np.empty((self.n_estimators, self.max_samples_, n_features))
        self._radius = np.empty((self.n_estimators, self.max_samples_))
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            centroid_index = rnd.choice(n_samples, self.max_samples_, replace=False)
            self._centroids[i] = X[centroid_index]
            distances, _indices = ArgKmin.compute(
                X=self._centroids[i],
                Y=self._centroids[i],
                k=2,
                metric="sqeuclidean",
                metric_kwargs={},
                strategy="auto",
                return_distance=True,
            )
            # column 0 = distance to self (always 0.0); column 1 = nearest other centroid
            self._radius[i] = distances[:, 1] * self.r

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        weighting = str(self.weighting).lower()
        if weighting not in {"binary", "boundary"}:
            raise ValueError("weighting must be one of {'binary', 'boundary'}")

        n, _m = X.shape
        blocks = []
        for i in range(self.n_estimators):
            distances = pairwise_distances(X, self._centroids[i], metric="sqeuclidean")
            inside = distances <= self._radius[i]

            if self.inclusive:
                # Mark all hyperspheres that contain each point.
                ik_value = np.zeros_like(distances, dtype=float)
                if weighting == "binary":
                    ik_value[inside] = 1.0
                else:
                    # Boundary-aware weights: higher near centroid, 0 at sphere boundary.
                    radius = np.maximum(self._radius[i], MIN_FLOAT)
                    boundary_weights = 1.0 - (distances / radius)
                    ik_value = np.where(inside, np.maximum(boundary_weights, 0.0), 0.0)

                # Normalize non-empty rows so each estimator block has unit L2 norm.
                weight_sum = np.linalg.norm(ik_value, axis=1, keepdims=True)
                ik_value = np.divide(
                    ik_value,
                    weight_sum,
                    out=np.zeros_like(ik_value),
                    where=weight_sum > 0,
                )
            else:
                ik_value = np.zeros_like(distances, dtype=float)
                masked_distances = np.where(inside, distances, np.inf)
                covered_rows = np.any(inside, axis=1)

                # Keep only one hypersphere: closest centroid among covering ones.
                if np.any(covered_rows):
                    nearest = np.argmin(masked_distances[covered_rows], axis=1)
                    ik_value[covered_rows, nearest] = 1.0

            blocks.append(sparse.csr_matrix(ik_value))

        return sparse.hstack(blocks, format="csr")
