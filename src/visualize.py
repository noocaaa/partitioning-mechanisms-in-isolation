"""
src/visualize.py  —  Partition geometry & kernel visualizations
===============================================================
Functions to draw the actual partitioning mechanisms on 2D data:
  • Voronoi cells        (aNNE)
  • Hyperspheres         (iNNE)
  • Axis-parallel splits (iForest)
  • Oblique splits       (SCiForest)

Also: kernel matrices, phi feature maps, anomaly scores, side-by-side grids.

Usage:
    from src.visualize import plot_all_partitions, plot_full_report
    from src.partitions import get_partition

    part = get_partition('anne', n_estimators=50, max_samples=16)
    part.fit(X)
    plot_all_partitions(X, y, {'anne': part}, dataset_name='moons')
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import Voronoi, voronoi_plot_2d

warnings.filterwarnings('ignore')

PARTITION_COLORS = {
    'anne':      '#ff7f0e',   # orange
    'inne':      '#1f77b4',   # blue
    'iforest':   '#2ca02c',   # green
    'sciforest': '#d62728',   # red
}

PARTITION_NAMES_SHORT = {
    'anne':      'Voronoi',
    'inne':      'Hypersphere',
    'iforest':   'Axis-parallel',
    'sciforest': 'Random hyperplane',
}


# ══════════════════════════════════════════════════════════════════════════
# GEOMETRY PLOTTERS
# ══════════════════════════════════════════════════════════════════════════

def _data_scatter(ax, X, y, s=25, alpha=0.7, legend=True):
    """Scatter plot of data coloured by true label."""
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1],
                   label=f'Class {label}', edgecolors='k',
                   s=s, alpha=alpha, zorder=3)
    if legend:
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)


def _plot_voronoi(ax, X, partition):
    """Draw Voronoi cells from the first estimator's centroids."""
    model = partition._model
    if not hasattr(model, 'centroids_'):
        ax.text(0.5, 0.5, 'centroids_ not available',
                transform=ax.transAxes, ha='center', va='center')
        return
    C = model.centroids_[0]          # (max_samples, n_features)
    if C.shape[1] != 2:
        ax.text(0.5, 0.5, 'not 2D',
                transform=ax.transAxes, ha='center', va='center')
        return
    # scipy Voronoi needs at least 3 non-collinear points
    if len(C) >= 3:
        try:
            vor = Voronoi(C)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                            line_colors=PARTITION_COLORS['anne'],
                            line_alpha=0.5, line_width=1.0,
                            point_size=12, point_color='red')
        except Exception:
            pass
    ax.scatter(C[:, 0], C[:, 1], c='red', marker='x', s=60,
               linewidths=1.5, zorder=4, label='Centroids')


def _plot_hyperspheres(ax, X, partition):
    """Draw hyperspheres from the first estimator."""
    model = partition._model
    if not (hasattr(model, 'centroids_') and hasattr(model, 'radius_')):
        ax.text(0.5, 0.5, 'centroids_/radius_ not available',
                transform=ax.transAxes, ha='center', va='center')
        return
    C = model.centroids_[0]          # (max_samples, 2)
    R = model.radius_[0]             # (max_samples,)
    if C.shape[1] != 2:
        ax.text(0.5, 0.5, 'not 2D',
                transform=ax.transAxes, ha='center', va='center')
        return
    for c, r in zip(C, R):
        circ = Circle(c, r, fill=False, edgecolor=PARTITION_COLORS['inne'],
                      alpha=0.35, linewidth=1.0, zorder=2)
        ax.add_patch(circ)
    ax.scatter(C[:, 0], C[:, 1], c='blue', marker='+', s=60,
               linewidths=1.5, zorder=4, label='Centroids')


def _draw_iforest_rects(ax, tree, node, x_min, x_max, y_min, y_max, depth=0):
    """Recursively draw axis-parallel splits and leaf rectangles."""
    if tree.feature[node] == -2:          # leaf
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=True, facecolor='lightgray', alpha=0.08,
                         edgecolor=PARTITION_COLORS['iforest'],
                         linewidth=0.6, zorder=1)
        ax.add_patch(rect)
        return
    feat = tree.feature[node]
    thr = tree.threshold[node]
    if feat == 0:
        ax.plot([thr, thr], [y_min, y_max], '-',
                color=PARTITION_COLORS['iforest'], alpha=0.6, lw=1.2, zorder=2)
        _draw_iforest_rects(ax, tree, tree.children_left[node],
                            x_min, thr, y_min, y_max, depth + 1)
        _draw_iforest_rects(ax, tree, tree.children_right[node],
                            thr, x_max, y_min, y_max, depth + 1)
    else:
        ax.plot([x_min, x_max], [thr, thr], '-',
                color=PARTITION_COLORS['iforest'], alpha=0.6, lw=1.2, zorder=2)
        _draw_iforest_rects(ax, tree, tree.children_left[node],
                            x_min, x_max, y_min, thr, depth + 1)
        _draw_iforest_rects(ax, tree, tree.children_right[node],
                            x_min, x_max, thr, y_max, depth + 1)


def _plot_iforest_splits(ax, X, partition, tree_idx=0):
    """Draw the first tree's recursive axis-parallel splits."""
    tree = partition._model.estimators_[tree_idx].tree_
    pad = 0.05
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    _draw_iforest_rects(ax, tree, 0, x_min, x_max, y_min, y_max)


def _line_bbox_intersections(feat, coef, split, x_min, x_max, y_min, y_max):
    """Return (x1,y1,x2,y2) of the oblique line inside the bounding box."""
    # In 2D: c0*x[feat[0]] + c1*x[feat[1]] = split
    # We work in the original coordinate space; feat may be [0,1] or permuted.
    if len(feat) < 2:
        # Single-feature split → axis-aligned
        f0, c0 = feat[0], coef[0]
        val = split / c0 if abs(c0) > 1e-9 else 0
        if f0 == 0:
            return (val, y_min, val, y_max)
        else:
            return (x_min, val, x_max, val)

    # Use first two features of the split
    f0, f1 = feat[0], feat[1]
    c0, c1 = coef[0], coef[1]
    pts = []
    # Intersection with x = x_min
    if abs(c1) > 1e-9:
        v = (split - c0 * x_min) / c1
        if y_min <= v <= y_max:
            pts.append((x_min, v))
    # Intersection with x = x_max
    if abs(c1) > 1e-9:
        v = (split - c0 * x_max) / c1
        if y_min <= v <= y_max:
            pts.append((x_max, v))
    # Intersection with y = y_min
    if abs(c0) > 1e-9:
        v = (split - c1 * y_min) / c0
        if x_min <= v <= x_max:
            pts.append((v, y_min))
    # Intersection with y = y_max
    if abs(c0) > 1e-9:
        v = (split - c1 * y_max) / c0
        if x_min <= v <= x_max:
            pts.append((v, y_max))

    if len(pts) >= 2:
        return (pts[0][0], pts[0][1], pts[1][0], pts[1][1])
    return None


def _draw_sciforest_lines(ax, node, x_min, x_max, y_min, y_max):
    """Recursively draw oblique split lines."""
    if node.get('leaf'):
        return
    feat = node['feat']
    coef = node['coef']
    split = node['split']
    seg = _line_bbox_intersections(feat, coef, split, x_min, x_max, y_min, y_max)
    if seg:
        ax.plot([seg[0], seg[2]], [seg[1], seg[3]], '-',
                color=PARTITION_COLORS['sciforest'], alpha=0.5, lw=1.2, zorder=2)
    _draw_sciforest_lines(ax, node['left'],  x_min, x_max, y_min, y_max)
    _draw_sciforest_lines(ax, node['right'], x_min, x_max, y_min, y_max)


def _plot_sciforest_splits(ax, X, partition, tree_idx=0):
    """Draw the first tree's oblique split lines."""
    tree = partition._trees[tree_idx]._tree
    pad = 0.05
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    _draw_sciforest_lines(ax, tree, x_min, x_max, y_min, y_max)


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════

def plot_geometry_2d(ax, X, y, partition, method, title=None, show_legend=True):
    """
    Plot partition geometry on 2D data.

    Parameters
    ----------
    ax : matplotlib Axes
    X  : array (n, 2)
    y  : array (n,)
    partition : fitted partition object
    method    : 'anne' | 'inne' | 'iforest' | 'sciforest'
    title     : optional ax title
    """
    if X.shape[1] != 2:
        ax.text(0.5, 0.5, 'Geometry plot requires 2D data',
                transform=ax.transAxes, ha='center', va='center')
        return

    _data_scatter(ax, X, y, legend=show_legend)

    if method == 'anne':
        _plot_voronoi(ax, X, partition)
    elif method == 'inne':
        _plot_hyperspheres(ax, X, partition)
    elif method == 'iforest':
        _plot_iforest_splits(ax, X, partition)
    elif method == 'sciforest':
        _plot_sciforest_splits(ax, X, partition)

    ax.set_xlim(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1)
    ax.set_ylim(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title or PARTITION_NAMES_SHORT.get(method, method), fontsize=10)
    ax.set_xlabel('x₁', fontsize=8)
    ax.set_ylabel('x₂', fontsize=8)


def plot_kernel_heatmap(ax, K, y, title='Kernel matrix', cmap='viridis'):
    """
    Plot a kernel matrix heatmap with rows/columns sorted by class label.
    """
    order = np.argsort(y)
    Ks = K[np.ix_(order, order)]
    im = ax.imshow(Ks, aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('samples (sorted by class)', fontsize=8)
    ax.set_ylabel('samples (sorted by class)', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Draw class boundary lines
    boundaries = np.where(np.diff(y[order]) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color='white', lw=0.8)
        ax.axvline(b, color='white', lw=0.8)


def plot_phi_spy(ax, phi, title='Φ feature map'):
    """
    Spy plot of the sparse binary feature map.
    """
    if hasattr(phi, 'toarray'):
        phi_dense = phi.toarray()
    else:
        phi_dense = np.asarray(phi)
    ax.imshow(phi_dense[:min(200, len(phi_dense)), :],
              aspect='auto', cmap='Greys', interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('leaf / cell index', fontsize=8)
    ax.set_ylabel('sample index', fontsize=8)


def plot_scores_2d(ax, X, y, scores, title='Anomaly scores'):
    """
    Scatter plot coloured by anomaly score (higher = more anomalous).
    """
    sc = ax.scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlBu_r',
                    edgecolors='k', s=30, alpha=0.8, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='score')


def plot_all_partitions(X, y, partitions, dataset_name='',
                        figsize=(12, 10), save_path=None):
    """
    Side-by-side 2×2 grid showing all 4 partition geometries on the same data.

    partitions : dict {'anne': part_obj, 'inne': part_obj, ...}
    """
    if X.shape[1] != 2:
        raise ValueError('plot_all_partitions requires 2D data')

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    for ax, (method, part) in zip(axes, partitions.items()):
        plot_geometry_2d(ax, X, y, part, method,
                         title=PARTITION_NAMES_SHORT.get(method, method),
                         show_legend=(method == 'anne'))

    fig.suptitle(f'{dataset_name} — Partition geometries', fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  Saved → {save_path}')
    return fig


def plot_full_report(X, y, partitions, dataset_name='',
                     kernels=None, scores=None, figsize=(14, 10),
                     save_path=None):
    """
    Comprehensive figure:
      row 0: 4 partition geometries
      row 1: IK kernel heatmaps for each partition
      row 2: IDK kernel heatmaps + anomaly scores (if available)

    partitions : dict {'anne': part_obj, ...}
    kernels    : optional dict {'anne': {'ik': K, 'idk': K}, ...}
    scores     : optional dict {'anne': scores_array, ...}
    """
    n_methods = len(partitions)
    if n_methods == 0:
        return None

    # Determine layout
    rows = 1 + (1 if kernels else 0) + (1 if scores else 0)
    fig, axes = plt.subplots(rows, n_methods, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = np.atleast_2d(axes)

    # Row 0: geometries
    for col, (method, part) in enumerate(partitions.items()):
        plot_geometry_2d(axes[0, col], X, y, part, method,
                         title=PARTITION_NAMES_SHORT.get(method, method),
                         show_legend=(col == 0))

    # Row 1: IK kernels
    if kernels:
        for col, method in enumerate(partitions.keys()):
            K = kernels.get(method, {}).get('ik')
            if K is not None:
                plot_kernel_heatmap(axes[1, col], K, y,
                                    title=f'IK — {method}')
            else:
                axes[1, col].axis('off')

    # Row 2: IDK kernels or scores
    if scores:
        for col, method in enumerate(partitions.keys()):
            s = scores.get(method)
            if s is not None and X.shape[1] == 2:
                plot_scores_2d(axes[-1, col], X, y, s,
                               title=f'Scores — {method}')
            else:
                axes[-1, col].axis('off')
    elif kernels:
        for col, method in enumerate(partitions.keys()):
            K = kernels.get(method, {}).get('idk')
            if K is not None:
                plot_kernel_heatmap(axes[-1, col], K, y,
                                    title=f'IDK — {method}')
            else:
                axes[-1, col].axis('off')

    fig.suptitle(f'{dataset_name} — Full report', fontsize=12, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f'  Saved → {save_path}')
    return fig


# ══════════════════════════════════════════════════════════════════════════
# STAND-ALONE HELPER
# ══════════════════════════════════════════════════════════════════════════

def fit_and_plot_all(X, y, dataset_name='', n_estimators=50, max_samples=16,
                     random_state=42, out_dir='results/figures'):
    """
    One-liner: fit all 4 partitions on (X,y) and save geometry + report figures.
    Only works for 2D data (geometry); kernel/phi plots work for any dimension.
    """
    from src.partitions import get_partition

    partitions = {}
    kernels = {}
    scores = {}

    for method in ['anne', 'inne', 'iforest', 'sciforest']:
        print(f'  Fitting {method} ...', end=' ')
        part = get_partition(method, n_estimators=n_estimators,
                             max_samples=max_samples, random_state=random_state)
        part.fit(X)
        partitions[method] = part
        kernels[method] = {
            'ik':  part.similarity_ik(X),
            'idk': part.similarity_idk(X),
        }
        scores[method] = part.idk_scores(X)
        print('OK')

    # Geometry comparison
    if X.shape[1] == 2:
        plot_all_partitions(
            X, y, partitions, dataset_name=dataset_name,
            save_path=os.path.join(out_dir, f'{dataset_name}_geometries.png')
        )

    # Full report
    plot_full_report(
        X, y, partitions, dataset_name=dataset_name,
        kernels=kernels, scores=scores,
        save_path=os.path.join(out_dir, f'{dataset_name}_report.png')
    )

    # Individual phi maps
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (method, part) in zip(axes.ravel(), partitions.items()):
        phi = part.transform(X)
        plot_phi_spy(ax, phi, title=f'Φ — {method}')
    fig.suptitle(f'{dataset_name} — Feature maps', fontsize=12)
    plt.tight_layout()
    phi_path = os.path.join(out_dir, f'{dataset_name}_phi_maps.png')
    fig.savefig(phi_path, dpi=200, bbox_inches='tight')
    print(f'  Saved → {phi_path}')
    plt.close('all')
