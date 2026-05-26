"""
notebooks/visualize_inne_space.py
==================================
Visualize how iNNE partitions 2-D space into hyperspheres for a synthetic
3-cluster dataset.

Each point in the background grid is coloured by the hypersphere it is
assigned to (nearest covering centroid).  Grid cells not covered by any
hypersphere are shown in light grey.  The hypersphere circles are drawn on
top, and the training points are overlaid.

The r parameter (0 < r ≤ 1) scales every centroid's radius:
    radius_i = r * dist(centroid_i, its_nearest_other_centroid)
A smaller r shrinks the spheres and increases uncovered space.

Usage:
    uv run python notebooks/visualize_inne_space.py
    uv run python notebooks/visualize_inne_space.py --r 0.5
    uv run python notebooks/visualize_inne_space.py --r 0.9 --max_samples 10 --seed 3
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import warnings

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IK_INNE = importlib.import_module("src._ik_inne").IK_INNE
IK_ANNE = importlib.import_module("src._ik_anne").IK_ANNE
get_partition = importlib.import_module("src.partitions").get_partition


FIGURE_PATH = os.path.join(
    ROOT, "figures", "anomaly_detection", "inne_space_partition.png"
)
FIGURE_PATH_ANNE = os.path.join(
    ROOT, "figures", "anomaly_detection", "anne_space_partition.png"
)
FIGURE_PATH_PROGRESS = os.path.join(
    ROOT, "figures", "anomaly_detection", "inne_space_progression.png"
)
ANIMATION_PATH_GIF = os.path.join(
    ROOT, "figures", "anomaly_detection", "inne_space_progression.gif"
)
ANIMATION_PATH_MP4 = os.path.join(
    ROOT, "figures", "anomaly_detection", "inne_space_progression.mp4"
)

# ── Colour palette ──────────────────────────────────────────────────────────

# Qualitative palette with enough distinct colours for up to ~20 centroids.
_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

UNCOVERED_COLOR = "#eeeeee"  # light grey for uncovered background
CLUSTER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]  # point colours by cluster


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)


def format_step_label(value: float) -> str:
    value = float(value)
    if value > 100:
        exponent = np.log10(value)
        rounded_exponent = round(exponent)
        if np.isclose(exponent, rounded_exponent, atol=1e-9):
            return rf"$10^{{{int(rounded_exponent)}}}$"

    rounded = round(value, 1)
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.1f}"


# ── Data generation ─────────────────────────────────────────────────────────


def make_data_with_anomalies(
    n_per_cluster: int = 60,
    n_anomalies: int = 15,
    random_state: int = 0,
    anomaly_style: str = "contextual",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_test, y_test) for anomaly/clustering eval.

    X_train contains only normal points from a mixed-shape 3-cluster layout.
    y_train are the corresponding cluster labels for clustering ARI.
    X_test is held-out normal points plus synthetic anomalies.
    y_test is 0 (normal) and 1 (anomaly).
    """
    X_train, y_train, X_test_normal = _make_challenging_clusters(
        n_per_cluster=n_per_cluster,
        n_holdout=20,
        random_state=random_state,
    )
    n_test_per_cluster = len(X_test_normal) // 3
    y_test_normal = np.concatenate(
        [
            np.zeros(n_test_per_cluster, dtype=np.int32),
            np.ones(n_test_per_cluster, dtype=np.int32),
            np.full(n_test_per_cluster, 2, dtype=np.int32),
        ]
    )
    X_ref = np.vstack([X_train, X_test_normal])
    y_ref = np.concatenate([y_train, y_test_normal])

    rng = np.random.default_rng(random_state + 1)
    if anomaly_style == "outside":
        x_anom = _sample_anomalies_outside_support(
            X_ref=X_ref,
            n_anomalies=n_anomalies,
            rng=rng,
            min_dist=0.55,
        )
    elif anomaly_style == "mixed":
        n_ctx = max(1, int(0.6 * n_anomalies))
        n_out = max(1, n_anomalies - n_ctx)
        x_ctx = _sample_contextual_anomalies(
            X_ref=X_ref,
            y_ref=y_ref,
            n_anomalies=n_ctx,
            rng=rng,
        )
        x_out = _sample_anomalies_outside_support(
            X_ref=X_ref,
            n_anomalies=n_out,
            rng=rng,
            min_dist=0.45,
        )
        x_anom = np.vstack([x_ctx, x_out]).astype(np.float32)
    else:
        x_anom = _sample_contextual_anomalies(
            X_ref=X_ref,
            y_ref=y_ref,
            n_anomalies=n_anomalies,
            rng=rng,
        )

    X_test = np.vstack([X_test_normal, x_anom]).astype(np.float32)
    y_test = np.concatenate(
        [
            np.zeros(len(X_test_normal), dtype=np.int32),
            np.ones(len(x_anom), dtype=np.int32),
        ]
    )
    return X_train, y_train, X_test, y_test


def _make_challenging_clusters(
    n_per_cluster: int,
    n_holdout: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a harder 2D dataset with mixed geometry and partial overlap."""
    rng = np.random.default_rng(random_state)

    n_total = n_per_cluster + n_holdout

    # Cluster 0: elongated, rotated Gaussian
    theta = np.deg2rad(35.0)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    base0 = rng.normal(loc=0.0, scale=[0.95, 0.20], size=(n_total, 2)).astype(
        np.float32
    )
    c0 = base0 @ rot.T + np.array([-2.2, -1.1], dtype=np.float32)

    # Cluster 1: crescent/arc (placed near the ring for intentional overlap)
    angle = rng.uniform(-2.3, 0.35, size=n_total)
    radius = rng.normal(loc=1.18, scale=0.11, size=n_total)
    c1 = np.column_stack(
        [1.8 + radius * np.cos(angle), 0.55 + 0.78 * radius * np.sin(angle)]
    ).astype(np.float32)
    c1 += rng.normal(0.0, [0.08, 0.06], size=(n_total, 2)).astype(np.float32)

    # Cluster 2: annulus/ring with empty inside.
    # Radial jitter is narrow so the center stays mostly empty.
    ring_angle = rng.uniform(0.0, 2.0 * np.pi, size=n_total)
    ring_radius = rng.normal(loc=1.00, scale=0.07, size=n_total)
    ring_radius = np.clip(ring_radius, 0.82, 1.18)
    c2 = np.column_stack(
        [
            2.45 + ring_radius * np.cos(ring_angle),
            -0.70 + ring_radius * np.sin(ring_angle),
        ]
    ).astype(np.float32)
    c2 += rng.normal(0.0, [0.03, 0.03], size=(n_total, 2)).astype(np.float32)

    train_parts = [c0[:n_per_cluster], c1[:n_per_cluster], c2[:n_per_cluster]]
    test_parts = [c0[n_per_cluster:], c1[n_per_cluster:], c2[n_per_cluster:]]

    X_train = np.vstack(train_parts).astype(np.float32)
    y_train = np.concatenate(
        [
            np.zeros(n_per_cluster, dtype=np.int32),
            np.ones(n_per_cluster, dtype=np.int32),
            np.full(n_per_cluster, 2, dtype=np.int32),
        ]
    )
    X_test_normal = np.vstack(test_parts).astype(np.float32)
    return X_train, y_train, X_test_normal


def _sample_anomalies_outside_support(
    X_ref: np.ndarray,
    n_anomalies: int,
    rng: np.random.Generator,
    min_dist: float = 0.55,
) -> np.ndarray:
    """Sample anomalies away from normal support by nearest-neighbor rejection."""
    x_min, x_max = float(X_ref[:, 0].min()), float(X_ref[:, 0].max())
    y_min, y_max = float(X_ref[:, 1].min()), float(X_ref[:, 1].max())
    pad = 1.0

    collected = []
    target = int(max(1, n_anomalies))
    max_rounds = 60
    for _ in range(max_rounds):
        if len(collected) >= target:
            break
        need = target - len(collected)
        cand = rng.uniform(
            low=[x_min - pad, y_min - pad],
            high=[x_max + pad, y_max + pad],
            size=(max(need * 5, 20), 2),
        ).astype(np.float32)
        diff = cand[:, None, :] - X_ref[None, :, :]
        min_d = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        keep = cand[min_d >= min_dist]
        if len(keep):
            collected.extend(list(keep[:need]))

    if not collected:
        # Fallback: still return a few broad-range outliers.
        return rng.uniform(
            low=[x_min - 1.5, y_min - 1.5],
            high=[x_max + 1.5, y_max + 1.5],
            size=(max(5, target), 2),
        ).astype(np.float32)

    return np.asarray(collected[:target], dtype=np.float32)


def _sample_contextual_anomalies(
    X_ref: np.ndarray,
    y_ref: np.ndarray,
    n_anomalies: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample hard anomalies close to data support (less trivially separable)."""
    target = int(max(1, n_anomalies))

    # Known ring center from generator; points inside hole are hard anomalies.
    ring_center = np.array([2.45, -0.70], dtype=np.float32)

    # Label-wise means for boundary perturbations.
    labels = np.unique(y_ref)
    means = {lab: X_ref[y_ref == lab].mean(axis=0) for lab in labels}

    n_hole = max(1, int(0.4 * target))
    n_boundary = max(1, int(0.4 * target))
    n_bridge = max(0, target - n_hole - n_boundary)

    # 1) Inside-ring-hole anomalies.
    ang = rng.uniform(0.0, 2.0 * np.pi, size=n_hole)
    rad = rng.uniform(0.05, 0.48, size=n_hole)
    hole = np.column_stack(
        [ring_center[0] + rad * np.cos(ang), ring_center[1] + rad * np.sin(ang)]
    ).astype(np.float32)
    hole += rng.normal(0.0, 0.02, size=hole.shape).astype(np.float32)

    # 2) Near-boundary anomalies from outward perturbations of normal points.
    idx = rng.choice(len(X_ref), size=n_boundary, replace=True)
    base = X_ref[idx]
    yb = y_ref[idx]
    direction = base - np.vstack([means[int(l)] for l in yb])
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.divide(direction, np.maximum(norm, 1e-8))
    step = rng.uniform(0.12, 0.38, size=(n_boundary, 1)).astype(np.float32)
    boundary = base + direction * step
    boundary += rng.normal(0.0, 0.03, size=boundary.shape).astype(np.float32)

    # 3) Bridge/overlap-region anomalies between crescent and ring area.
    if n_bridge > 0:
        cov = np.array([[0.08, 0.02], [0.02, 0.06]], dtype=np.float32)
        bridge = rng.multivariate_normal([1.95, -0.15], cov, size=n_bridge).astype(
            np.float32
        )
    else:
        bridge = np.empty((0, 2), dtype=np.float32)

    cand = np.vstack([hole, boundary, bridge]).astype(np.float32)

    # Keep anomalies in a "hard" distance band relative to normal support.
    diff = cand[:, None, :] - X_ref[None, :, :]
    min_d = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
    keep = (min_d >= 0.08) & (min_d <= 0.75)
    kept = cand[keep]

    if len(kept) >= target:
        return kept[:target]

    # Fallback: top-up with outside anomalies not too far away.
    topup = _sample_anomalies_outside_support(
        X_ref=X_ref,
        n_anomalies=target - len(kept),
        rng=rng,
        min_dist=0.28,
    )
    return np.vstack([kept, topup]).astype(np.float32)


def make_three_clusters(
    n_per_cluster: int = 60, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, labels) for three challenging, differently shaped clusters."""
    X, y, _ = _make_challenging_clusters(
        n_per_cluster=n_per_cluster,
        n_holdout=0,
        random_state=random_state,
    )
    return X, y


# ── Background partition colouring ─────────────────────────────────────────


def assign_grid(
    grid_xy: np.ndarray,
    centroids: np.ndarray,
    radii_sq: np.ndarray,
) -> np.ndarray:
    """Return centroid assignment index for each grid point (-1 = uncovered).

    Parameters
    ----------
    grid_xy   : (N, 2) – grid points to classify
    centroids : (K, 2) – centroid positions for one estimator
    radii_sq  : (K,)   – squared radii for each centroid
    """
    # Squared Euclidean distances: (N, K)
    diff = grid_xy[:, None, :] - centroids[None, :, :]  # (N, K, 2)
    dist_sq = (diff**2).sum(axis=2)  # (N, K)

    inside = dist_sq <= radii_sq[None, :]  # (N, K) bool
    masked = np.where(inside, dist_sq, np.inf)  # inf for outside

    assignment = np.argmin(masked, axis=1)  # (N,)
    covered = inside.any(axis=1)
    assignment[~covered] = -1  # mark uncovered
    return assignment


def grid_extent(X: np.ndarray, pad: float = 0.6) -> tuple[float, float, float, float]:
    return (
        float(X[:, 0].min() - pad),
        float(X[:, 0].max() + pad),
        float(X[:, 1].min() - pad),
        float(X[:, 1].max() + pad),
    )


def build_grid(
    X: np.ndarray, grid_resolution: int = 600, pad: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    x_min, x_max, y_min, y_max = grid_extent(X, pad=pad)
    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    return grid_pts, xx, yy, [x_min, x_max, y_min, y_max]


def inne_grid_stats(
    grid_xy: np.ndarray,
    centroids: np.ndarray,
    radii_sq: np.ndarray,
) -> dict[str, float]:
    diff = grid_xy[:, None, :] - centroids[None, :, :]
    dist_sq = (diff**2).sum(axis=2)
    inside = dist_sq <= radii_sq[None, :]
    covered = inside.any(axis=1)
    return {
        "coverage": float(covered.mean()),
        "mean_overlap": float(inside.sum(axis=1).mean()),
        "max_overlap": float(inside.sum(axis=1).max()),
    }


def compute_anomaly_auc_ik_for_r_range(
    X: np.ndarray,
    y: np.ndarray,
    r_values: list[float],
    n_estimators: int,
    max_samples: int,
    random_state: int,
    n_runs: int = 5,
) -> list[float]:
    """Return mean AUC vs r using run_anomaly IK scoring across runs."""
    aucs = []
    for r_val in r_values:
        run_aucs = []
        for run in range(max(1, n_runs)):
            part = get_partition(
                "inne",
                kernel="ik",
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=random_state + run,
                r=r_val,
            )
            part.fit(X)
            K = part.similarity_ik(X)
            scores = 1.0 - K.mean(axis=1)
            try:
                run_aucs.append(float(roc_auc_score(y, scores)))
            except ValueError:
                run_aucs.append(float("nan"))
        aucs.append(float(np.nanmean(run_aucs)))
    return aucs


def compute_clustering_ari_ik_for_r_range(
    X: np.ndarray,
    y: np.ndarray,
    r_values: list[float],
    n_estimators: int,
    max_samples: int,
    random_state: int,
    n_runs: int = 5,
) -> list[float]:
    """Return mean ARI vs r using run_clustering IK + spectral flow across runs."""
    n_clusters = len(np.unique(y))
    aris = []
    for r_val in r_values:
        run_aris = []
        for run in range(max(1, n_runs)):
            part = get_partition(
                "inne",
                kernel="ik",
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=random_state + run,
                r=r_val,
            )
            part.fit(X)
            K = part.similarity_ik(X)
            K = np.clip((K + K.T) / 2.0, 0.0, 1.0)
            np.fill_diagonal(K, 1.0)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding may not work as expected.",
                    category=UserWarning,
                )
                labels = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    random_state=random_state + run,
                    n_init=10,
                ).fit_predict(K)
            run_aris.append(float(adjusted_rand_score(y, labels)))
        aris.append(float(np.nanmean(run_aris)))
    return aris


# ── Plotting ────────────────────────────────────────────────────────────────

# ── aNNE Voronoi partition ──────────────────────────────────────────────────


def plot_anne_space(
    X: np.ndarray,
    y: np.ndarray,
    model: IK_ANNE,
    estimator_idx: int = 0,
    grid_resolution: int = 600,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw the aNNE Voronoi partition for one estimator on *ax*.

    Each grid point is coloured by its nearest centroid — there is no radius
    cutoff, so every grid point is always assigned to exactly one cell.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    # Recover the centroid coordinates for this estimator
    id_to_index = {id_val: idx for idx, id_val in enumerate(model.unique_ids)}
    center_ids = model.center_ids[estimator_idx]  # (K,) indices into X
    center_indices = np.array([id_to_index[c] for c in center_ids])
    centroids = model.center_data[center_indices]  # (K, 2)

    K = centroids.shape[0]
    palette = (_PALETTE * ((K // len(_PALETTE)) + 1))[:K]

    grid_pts, _xx, _yy, extent = build_grid(X, grid_resolution=grid_resolution)
    x_min, x_max, y_min, y_max = extent

    # Nearest-centroid assignment (pure Voronoi — no radius cutoff)
    diff = grid_pts[:, None, :] - centroids[None, :, :]  # (N, K, 2)
    dist_sq = (diff**2).sum(axis=2)  # (N, K)
    assignment = np.argmin(dist_sq, axis=1)  # (N,)  always assigned

    rgb_map = np.stack([hex_to_rgb(c) for c in palette])  # (K, 3)
    img_rgb = rgb_map[assignment].reshape(grid_resolution, grid_resolution, 3)

    ax.imshow(
        img_rgb,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        alpha=0.40,
    )

    # Mark centroids
    for k in range(K):
        ax.plot(
            centroids[k, 0],
            centroids[k, 1],
            marker="+",
            markersize=7,
            color=palette[k],
            markeredgewidth=1.2,
            zorder=4,
        )

    # Overlay training points coloured by cluster
    for cluster_id in np.unique(y):
        mask = y == cluster_id
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=20,
            color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title or f"aNNE Voronoi partition (estimator {estimator_idx})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    return ax


def plot_inne_space(
    X: np.ndarray,
    y: np.ndarray,
    model: IK_INNE,
    estimator_idx: int = 0,
    r: float = 0.9,
    grid_resolution: int = 600,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw the iNNE hypersphere partition for one estimator on *ax*."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    centroids = model._centroids[estimator_idx]  # (K, 2)
    base_radii_sq = model._radius[estimator_idx]
    radii_sq = base_radii_sq * (r / model.r)

    return plot_inne_space_from_geometry(
        X,
        y,
        centroids,
        radii_sq,
        estimator_idx=estimator_idx,
        r=r,
        grid_resolution=grid_resolution,
        ax=ax,
        title=title,
    )


def plot_inne_space_from_geometry(
    X: np.ndarray,
    y: np.ndarray,
    centroids: np.ndarray,
    radii_sq: np.ndarray,
    estimator_idx: int = 0,
    r: float = 1.0,
    grid_resolution: int = 600,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Draw an iNNE partition from explicit centroid and radius geometry."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    grid_pts, _xx, _yy, extent = build_grid(X, grid_resolution=grid_resolution)
    x_min, x_max, y_min, y_max = extent

    # Assign each grid point to a hypersphere
    assignment = assign_grid(grid_pts, centroids, radii_sq)

    # Build RGBA image
    K = centroids.shape[0]
    palette = (_PALETTE * ((K // len(_PALETTE)) + 1))[:K]
    uncov_rgb = np.array([0xEE, 0xEE, 0xEE], dtype=np.uint8)

    rgb_map = np.stack([hex_to_rgb(c) for c in palette])  # (K, 3)

    img_rgb = np.where(
        (assignment >= 0)[:, None],
        rgb_map[np.maximum(assignment, 0)],
        uncov_rgb[None, :],
    ).reshape(grid_resolution, grid_resolution, 3)

    ax.imshow(
        img_rgb,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
        alpha=0.45,
    )

    # Draw hypersphere circles
    for k in range(K):
        radius = float(np.sqrt(radii_sq[k]))
        circle = Circle(
            (centroids[k, 0], centroids[k, 1]),
            radius,
            linewidth=0.9,
            edgecolor=palette[k],
            facecolor="none",
            alpha=0.75,
            zorder=3,
        )
        ax.add_patch(circle)
        # Mark centroid
        ax.plot(
            centroids[k, 0],
            centroids[k, 1],
            marker="+",
            markersize=7,
            color=palette[k],
            markeredgewidth=1.2,
            zorder=4,
        )

    # Overlay training points coloured by cluster
    for cluster_id in np.unique(y):
        mask = y == cluster_id
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=20,
            color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title or f"iNNE partition (estimator {estimator_idx},  r={r})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    return ax


def plot_inne_r_progression(
    X: np.ndarray,
    y: np.ndarray,
    model: IK_INNE,
    estimator_idx: int,
    r_values: list[float],
    grid_resolution: int = 350,
) -> plt.Figure:
    """Show fixed-centroid iNNE partitions as radii expand with r."""
    centroids = model._centroids[estimator_idx]
    base_radii_sq = model._radius[estimator_idx] / model.r
    grid_pts, _xx, _yy, _extent = build_grid(X, grid_resolution=grid_resolution)

    fig = plt.figure(figsize=(4.2 * len(r_values), 8.5))
    gs = fig.add_gridspec(2, len(r_values), height_ratios=[4.2, 1.4])
    metric_ax = fig.add_subplot(gs[1, :])

    coverages = []
    overlaps = []
    x_pos = np.arange(len(r_values), dtype=np.int32)
    x_labels = [format_step_label(rv) for rv in r_values]
    for idx, r_value in enumerate(r_values):
        ax = fig.add_subplot(gs[0, idx])
        radii_sq = base_radii_sq * r_value
        stats = inne_grid_stats(grid_pts, centroids, radii_sq)
        coverages.append(stats["coverage"])
        overlaps.append(stats["mean_overlap"])
        plot_inne_space_from_geometry(
            X,
            y,
            centroids,
            radii_sq,
            estimator_idx=estimator_idx,
            r=r_value,
            grid_resolution=grid_resolution,
            ax=ax,
            title=f"r = {r_value:.2f}",
        )
        ax.text(
            0.02,
            0.02,
            f"covered={stats['coverage']:.2%}\nmean overlap={stats['mean_overlap']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    metric_ax.plot(x_pos, coverages, marker="o", linewidth=1.8, label="Coverage")
    metric_ax.plot(
        x_pos,
        overlaps,
        marker="s",
        linewidth=1.8,
        label="Mean overlap",
    )
    metric_ax.set_xlabel("r (uniformly spaced steps)")
    metric_ax.set_ylabel("Grid statistic")
    metric_ax.set_title("Fixed-estimator progression as r expands")
    metric_ax.set_xticks(x_pos)
    metric_ax.set_xticklabels(x_labels)
    metric_ax.grid(alpha=0.25)
    metric_ax.legend(framealpha=0.8)
    fig.suptitle(
        f"iNNE radius progression  (estimator {estimator_idx}, max_samples={model.max_samples_})",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def save_inne_r_animation(
    X: np.ndarray,
    y: np.ndarray,
    model: IK_INNE,
    estimator_idx: int,
    r_values: list[float],
    output_path: str,
    X_anom: np.ndarray | None = None,
    y_anom: np.ndarray | None = None,
    X_cluster: np.ndarray | None = None,
    y_cluster: np.ndarray | None = None,
    X_anomaly_points: np.ndarray | None = None,
    metric_n_estimators: int = 200,
    metric_n_runs: int = 5,
    fps: int = 3,
    grid_resolution: int = 350,
) -> str:
    """Save a fixed-estimator iNNE radius progression as GIF or MP4.

    The top panel animates the hypersphere partition as r increases.
    The middle panel shows static AUC/ARI-vs-r curves with moving markers.
    The bottom panel shows static coverage and mean-overlap curves.
    """
    centroids = model._centroids[estimator_idx]
    base_radii_sq = model._radius[estimator_idx] / model.r
    grid_pts, _xx, _yy, _extent = build_grid(X, grid_resolution=grid_resolution)

    # Pre-compute all static curves
    if X_anom is not None and y_anom is not None:
        auc_values = compute_anomaly_auc_ik_for_r_range(
            X_anom,
            y_anom,
            r_values,
            n_estimators=metric_n_estimators,
            max_samples=model.max_samples_,
            random_state=model.random_state,
            n_runs=metric_n_runs,
        )
    else:
        auc_values = [float("nan")] * len(r_values)

    if X_cluster is not None and y_cluster is not None:
        ari_values = compute_clustering_ari_ik_for_r_range(
            X_cluster,
            y_cluster,
            r_values,
            n_estimators=metric_n_estimators,
            max_samples=model.max_samples_,
            random_state=model.random_state,
            n_runs=metric_n_runs,
        )
    else:
        ari_values = [float("nan")] * len(r_values)

    coverage_values = []
    overlap_values = []
    x_pos = np.arange(len(r_values), dtype=np.int32)
    x_labels = [format_step_label(rv) for rv in r_values]
    for r_val in r_values:
        s = inne_grid_stats(grid_pts, centroids, base_radii_sq * r_val)
        coverage_values.append(s["coverage"])
        overlap_values.append(s["mean_overlap"])

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], hspace=0.42, wspace=0.32)
    space_ax = fig.add_subplot(gs[:, 0])  # left column — full height
    auc_ax = fig.add_subplot(gs[0, 1])  # right column — top
    stats_ax = fig.add_subplot(gs[1, 1])  # right column — bottom

    def draw_frame(frame_idx: int):
        r_value = r_values[frame_idx]
        radii_sq = base_radii_sq * r_value

        space_ax.clear()
        auc_ax.clear()
        stats_ax.clear()

        plot_inne_space_from_geometry(
            X,
            y,
            centroids,
            radii_sq,
            estimator_idx=estimator_idx,
            r=r_value,
            grid_resolution=grid_resolution,
            ax=space_ax,
            title=f"iNNE partition at r = {r_value:.2f}",
        )
        if X_anomaly_points is not None and len(X_anomaly_points) > 0:
            space_ax.scatter(
                X_anomaly_points[:, 0],
                X_anomaly_points[:, 1],
                s=34,
                color="#111111",
                marker="x",
                linewidths=1.2,
                zorder=6,
                label="Anomaly",
            )
            handles, labels = space_ax.get_legend_handles_labels()
            if "Anomaly" in labels:
                uniq = {}
                for h, lab in zip(handles, labels):
                    if lab not in uniq:
                        uniq[lab] = h
                space_ax.legend(
                    uniq.values(),
                    uniq.keys(),
                    loc="upper right",
                    fontsize=8,
                    framealpha=0.7,
                )
        stats = inne_grid_stats(grid_pts, centroids, radii_sq)
        space_ax.text(
            0.02,
            0.02,
            f"covered={stats['coverage']:.2%}\nmean overlap={stats['mean_overlap']:.2f}",
            transform=space_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

        # Middle panel: AUC + ARI
        auc_ax.plot(
            x_pos,
            auc_values,
            color="#2ca02c",
            marker="o",
            markersize=5,
            linewidth=1.8,
            label="AUC (anomaly, IK score)",
        )
        auc_ax.plot(
            x_pos,
            ari_values,
            color="#d62728",
            marker="s",
            markersize=5,
            linewidth=1.8,
            linestyle="--",
            label="ARI (clustering, IK)",
        )
        idx = frame_idx
        auc_ax.axvline(idx, color="#444444", linewidth=1.2, alpha=0.8)
        if not np.isnan(auc_values[idx]):
            auc_ax.scatter([idx], [auc_values[idx]], color="#2ca02c", s=60, zorder=4)
        if not np.isnan(ari_values[idx]):
            auc_ax.scatter(
                [idx], [ari_values[idx]], color="#d62728", marker="s", s=60, zorder=4
            )
        auc_ax.set_xlabel("r (uniformly spaced steps)")
        auc_ax.set_ylabel("Score")
        auc_ax.set_title(
            f"AUC and ARI vs r  (IK metrics: t={metric_n_estimators}, runs={metric_n_runs})"
        )
        auc_ax.set_xticks(x_pos)
        auc_ax.set_xticklabels(x_labels)
        auc_ax.grid(alpha=0.25)
        auc_ax.legend(fontsize=8, framealpha=0.8)

        valid_metric = [v for v in (auc_values + ari_values) if not np.isnan(v)]
        if valid_metric:
            auc_ax.set_ylim(
                max(0, min(valid_metric) - 0.05), min(1, max(valid_metric) + 0.05)
            )

        current_auc = auc_values[frame_idx]
        if not np.isnan(current_auc):
            auc_ax.text(
                0.02,
                0.96,
                f"AUC = {current_auc:.3f}\nARI = {ari_values[frame_idx]:.3f}",
                transform=auc_ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )

        # Bottom panel: coverage + mean overlap (twin y-axis)
        stats_ax.plot(
            x_pos,
            coverage_values,
            color="#1f77b4",
            marker="o",
            markersize=5,
            linewidth=1.8,
            label="Coverage",
        )
        stats_ax.scatter([idx], [coverage_values[idx]], color="#1f77b4", s=60, zorder=4)
        stats_ax.set_ylabel("Coverage", color="#1f77b4")
        stats_ax.tick_params(axis="y", labelcolor="#1f77b4")
        stats_ax.set_ylim(0, 1.05)

        twin_ax = stats_ax.twinx()
        twin_ax.plot(
            x_pos,
            overlap_values,
            color="#ff7f0e",
            marker="s",
            markersize=5,
            linewidth=1.8,
            linestyle="--",
            label="Mean overlap",
        )
        twin_ax.scatter(
            [idx], [overlap_values[idx]], color="#ff7f0e", marker="s", s=60, zorder=4
        )
        twin_ax.set_ylabel("Mean overlap", color="#ff7f0e")
        twin_ax.tick_params(axis="y", labelcolor="#ff7f0e")

        stats_ax.axvline(idx, color="#444444", linewidth=1.2, alpha=0.8)
        stats_ax.set_xlabel("r (uniformly spaced steps)")
        stats_ax.set_title("Coverage and mean overlap vs r")
        stats_ax.set_xticks(x_pos)
        stats_ax.set_xticklabels(x_labels)
        stats_ax.grid(alpha=0.25)

        # Combined legend
        lines1, labels1 = stats_ax.get_legend_handles_labels()
        lines2, labels2 = twin_ax.get_legend_handles_labels()
        stats_ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0.8)

        fig.suptitle(
            f"iNNE radius progression  (estimator {estimator_idx}, max_samples={model.max_samples_})",
            fontsize=13,
        )
        fig.subplots_adjust(top=0.90)

    anim = animation.FuncAnimation(
        fig,
        draw_frame,
        frames=len(r_values),
        interval=max(1, int(1000 / fps)),
        repeat=True,
    )

    suffix = os.path.splitext(output_path)[1].lower()
    if suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    elif suffix == ".mp4":
        if not animation.FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg is not available for MP4 export")
        writer = animation.FFMpegWriter(fps=fps)
    else:
        raise ValueError("animation output must end with .gif or .mp4")

    anim.save(output_path, writer=writer, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize iNNE space partition")
    parser.add_argument(
        "--r", type=float, default=1.0, help="Radius scaling factor (default: 1.0)"
    )
    parser.add_argument(
        "--mode",
        choices=["inne", "anne", "both", "progress", "animate"],
        default="inne",
        help="Which partition to visualize (default: inne)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=12,
        help="Centroids per estimator (default: 12)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1,
        help="Number of estimators to show (default: 1, max: 4)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--n_points", type=int, default=60, help="Points per cluster (default: 60)"
    )
    parser.add_argument(
        "--compare_r",
        action="store_true",
        help="Show a 2×2 grid comparing r ∈ {0.3, 0.6, 0.9, 1.2}",
    )
    parser.add_argument(
        "--progress_values",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
        help="r values for fixed-estimator progression mode",
    )
    parser.add_argument(
        "--animation_format",
        choices=["gif", "mp4"],
        default="gif",
        help="Output format for animate mode",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Frames per second for animate mode",
    )
    parser.add_argument(
        "--anomaly_style",
        choices=["outside", "mixed", "contextual"],
        default="contextual",
        help="Anomaly generation style for animate mode (default: contextual)",
    )
    parser.add_argument(
        "--metric_n_estimators",
        type=int,
        default=200,
        help="Number of estimators for IK metric curves (default: 200)",
    )
    parser.add_argument(
        "--metric_n_runs",
        type=int,
        default=5,
        help="Number of random runs to average for metric curves (default: 5)",
    )
    args = parser.parse_args()

    X, y = make_three_clusters(n_per_cluster=args.n_points, random_state=args.seed)

    if args.mode == "anne":
        n_show = min(args.n_estimators, 4)
        model = IK_ANNE(
            n_estimators=n_show,
            max_samples=args.max_samples,
            random_state=args.seed,
        )
        model.fit(X)
        if n_show == 1:
            fig, ax = plt.subplots(figsize=(7, 7))
            fig.suptitle(
                f"aNNE Voronoi partition  (max_samples={args.max_samples})",
                fontsize=13,
            )
            plot_anne_space(X, y, model, estimator_idx=0, ax=ax)
        else:
            ncols = 2
            nrows = (n_show + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows))
            fig.suptitle(
                f"aNNE Voronoi partitions  (max_samples={args.max_samples})",
                fontsize=13,
            )
            for i, ax in enumerate(np.array(axes).flat):
                if i < n_show:
                    plot_anne_space(
                        X, y, model, estimator_idx=i, ax=ax, title=f"Estimator {i}"
                    )
                else:
                    ax.set_visible(False)
        plt.tight_layout()
        save_path = FIGURE_PATH_ANNE
    elif args.mode == "progress":
        model = IK_INNE(
            n_estimators=1,
            max_samples=args.max_samples,
            random_state=args.seed,
            overlapping=False,
            r=1.0,
        )
        model.fit(X)
        fig = plot_inne_r_progression(
            X,
            y,
            model,
            estimator_idx=0,
            r_values=args.progress_values,
        )
        save_path = FIGURE_PATH_PROGRESS
    elif args.mode == "animate":
        X_train, y_train, X_test, y_test = make_data_with_anomalies(
            n_per_cluster=args.n_points,
            random_state=args.seed,
            anomaly_style=args.anomaly_style,
        )
        X_anom = np.vstack([X_train, X_test]).astype(np.float32)
        y_anom = np.concatenate(
            [np.zeros(len(X_train), dtype=np.int32), y_test.astype(np.int32)]
        )
        X_anomaly_points = X_anom[y_anom == 1]
        model = IK_INNE(
            n_estimators=1,
            max_samples=args.max_samples,
            random_state=args.seed,
            overlapping=False,
            r=1.0,
        )
        model.fit(X_train)
        save_path = (
            ANIMATION_PATH_GIF if args.animation_format == "gif" else ANIMATION_PATH_MP4
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        saved_path = save_inne_r_animation(
            X_train,
            y_train,
            model,
            estimator_idx=0,
            r_values=args.progress_values,
            output_path=save_path,
            X_anom=X_anom,
            y_anom=y_anom,
            X_cluster=X_train,
            y_cluster=y_train,
            X_anomaly_points=X_anomaly_points,
            metric_n_estimators=args.metric_n_estimators,
            metric_n_runs=args.metric_n_runs,
            fps=args.fps,
        )
        print(f"Saved → {saved_path}")
        return
    elif args.mode == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(
            f"Space partition comparison  (max_samples={args.max_samples})",
            fontsize=13,
        )
        anne_model = IK_ANNE(
            n_estimators=1, max_samples=args.max_samples, random_state=args.seed
        )
        anne_model.fit(X)
        plot_anne_space(
            X,
            y,
            anne_model,
            estimator_idx=0,
            ax=axes[0],
            title="aNNE — Voronoi (nearest centroid)",
        )
        inne_model = IK_INNE(
            n_estimators=1,
            max_samples=args.max_samples,
            random_state=args.seed,
            overlapping=False,
            r=args.r,
        )
        inne_model.fit(X)
        plot_inne_space(
            X,
            y,
            inne_model,
            estimator_idx=0,
            r=args.r,
            ax=axes[1],
            title=f"iNNE — Hyperspheres (r={args.r})",
        )
        plt.tight_layout()
        save_path = FIGURE_PATH
    elif args.compare_r:
        r_values = [0.3, 0.6, 0.9, 1.2]
        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        fig.suptitle("iNNE space partition — effect of radius scaling r", fontsize=14)
        for ax, r_val in zip(axes.flat, r_values):
            model = IK_INNE(
                n_estimators=1,
                max_samples=args.max_samples,
                random_state=args.seed,
                overlapping=False,
                r=r_val,
            )
            model.fit(X)
            plot_inne_space(
                X, y, model, estimator_idx=0, r=r_val, ax=ax, title=f"r = {r_val}"
            )
        plt.tight_layout()
        save_path = FIGURE_PATH
    else:  # inne (default)
        n_show = min(args.n_estimators, 4)
        model = IK_INNE(
            n_estimators=n_show,
            max_samples=args.max_samples,
            random_state=args.seed,
            overlapping=False,
            r=args.r,
        )
        model.fit(X)

        if n_show == 1:
            fig, ax = plt.subplots(figsize=(7, 7))
            fig.suptitle(
                f"iNNE hypersphere partition  (max_samples={args.max_samples}, r={args.r})",
                fontsize=13,
            )
            plot_inne_space(X, y, model, estimator_idx=0, r=args.r, ax=ax)
        else:
            ncols = 2
            nrows = (n_show + 1) // 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows))
            fig.suptitle(
                f"iNNE hypersphere partitions  (max_samples={args.max_samples}, r={args.r})",
                fontsize=13,
            )
            for i, ax in enumerate(np.array(axes).flat):
                if i < n_show:
                    plot_inne_space(
                        X,
                        y,
                        model,
                        estimator_idx=i,
                        r=args.r,
                        ax=ax,
                        title=f"Estimator {i}  (r={args.r})",
                    )
                else:
                    ax.set_visible(False)
        plt.tight_layout()
        save_path = FIGURE_PATH

    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    # plt.show()


if __name__ == "__main__":
    main()
