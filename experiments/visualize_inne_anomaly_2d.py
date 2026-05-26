"""
Create a synthetic 2D anomaly-detection dataset, fit IK_INNE, and visualize
one estimator's hyperspheres together with the sparse feature mapping phi.

Usage:
    uv run python experiments/visualize_inne_anomaly_2d.py
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IK_INNE = importlib.import_module("src._ik_inne").IK_INNE


FIGURE_PATH = os.path.join(ROOT, "figures", "anomaly_detection", "inne_2d_demo.png")
SCORES_PATH = os.path.join(
    ROOT, "results", "anomaly_detection", "inne_2d_demo_scores.csv"
)
MAPPINGS_PATH = os.path.join(
    ROOT, "results", "anomaly_detection", "inne_2d_demo_mappings.csv"
)


@dataclass
class DemoData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class DemoResult:
    name: str
    model: Any
    scores: np.ndarray
    phi_dense: np.ndarray
    auc: float


@dataclass
class MappingRow:
    variant: str
    label_name: str
    example_rank: int
    sample_index: int
    tag: str
    x: float
    y: float
    score: float
    active_features: int
    phi_active_indices: str
    phi_active_values: str


def make_demo_data(random_state: int = 7) -> DemoData:
    rng = np.random.default_rng(random_state)

    train_a = rng.normal(loc=(-1.25, -0.2), scale=(0.30, 0.22), size=(52, 2))
    train_b = rng.normal(loc=(1.15, 0.45), scale=(0.35, 0.25), size=(52, 2))
    X_train = np.vstack([train_a, train_b]).astype(np.float32)

    test_normal_a = rng.normal(loc=(-1.15, -0.15), scale=(0.34, 0.24), size=(20, 2))
    test_normal_b = rng.normal(loc=(1.10, 0.40), scale=(0.38, 0.27), size=(20, 2))
    anomalies = np.array(
        [
            (-2.35, 1.70),
            (-1.95, -1.70),
            (0.15, 2.25),
            (2.15, -1.45),
            (2.65, 1.65),
            (-2.70, 0.20),
            (0.05, -2.00),
            (2.95, 0.10),
        ],
        dtype=np.float32,
    )

    X_test = np.vstack([test_normal_a, test_normal_b, anomalies]).astype(np.float32)
    y_test = np.concatenate(
        [
            np.zeros(len(test_normal_a) + len(test_normal_b), dtype=np.int32),
            np.ones(len(anomalies), dtype=np.int32),
        ]
    )
    return DemoData(X_train=X_train, X_test=X_test, y_test=y_test)


def idk_anomaly_scores(model: Any, X_train: np.ndarray, X_test: np.ndarray):
    phi_train = model.transform(X_train).tocsr()
    phi_test = model.transform(X_test).tocsr()

    train_kme = np.asarray(phi_train.mean(axis=0)).ravel()
    raw_similarity = np.asarray(phi_test.dot(train_kme)).ravel() / model.n_estimators

    phi_sq = np.asarray(phi_test.power(2).sum(axis=1)).ravel()
    point_norm = np.sqrt(phi_sq / model.n_estimators)
    train_norm = np.sqrt(np.dot(train_kme, train_kme) / model.n_estimators)
    denom = point_norm * train_norm

    similarity = np.divide(
        raw_similarity,
        denom,
        out=np.zeros_like(raw_similarity),
        where=denom > 0,
    )
    scores = np.clip(1.0 - similarity, 0.0, 1.0)
    return scores, phi_test


def build_demo_result(
    name: str,
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> DemoResult:
    scores, phi_test = idk_anomaly_scores(model, X_train, X_test)
    phi_dense = phi_test.toarray()
    auc = float(roc_auc_score(y_test, scores))
    return DemoResult(
        name=name, model=model, scores=scores, phi_dense=phi_dense, auc=auc
    )


def save_scores(demo: DemoData, results: list[DemoResult]):
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    rows = []
    for result in results:
        phi_dense = result.phi_dense
        for idx, (point, label, score) in enumerate(
            zip(demo.X_test, demo.y_test, result.scores, strict=True)
        ):
            rows.append(
                {
                    "variant": result.name,
                    "sample_index": idx,
                    "x": point[0],
                    "y": point[1],
                    "label": label,
                    "score": score,
                    "active_features": int((phi_dense[idx] > 0).sum()),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(SCORES_PATH, index=False)


def _format_phi_mapping(phi_row: np.ndarray) -> tuple[str, str]:
    active_idx = np.flatnonzero(phi_row)
    active_values = phi_row[active_idx]
    indices_text = ";".join(str(int(idx)) for idx in active_idx)
    values_text = ";".join(f"{float(value):.4f}" for value in active_values)
    return indices_text, values_text


def _estimator_phi_block(
    model: Any, phi_row: np.ndarray, estimator_index: int
) -> np.ndarray:
    assert model._centroids is not None
    block_width = int(model._centroids.shape[1])
    start = estimator_index * block_width
    stop = start + block_width
    return phi_row[start:stop]


def _select_representative_indices(
    y_test: np.ndarray,
    scores: np.ndarray,
    label: int,
    count: int = 3,
) -> list[int]:
    label_idx = np.flatnonzero(y_test == label)
    if len(label_idx) == 0:
        return []

    ordered = label_idx[np.argsort(scores[label_idx])]
    anchors = np.linspace(0, len(ordered) - 1, num=min(count, len(ordered)), dtype=int)
    chosen = ordered[anchors]
    return [int(idx) for idx in chosen]


def _shared_representative_indices(
    result: DemoResult, y_test: np.ndarray
) -> dict[str, list[int]]:
    return {
        "normal": _select_representative_indices(y_test, result.scores, label=0),
        "anomaly": _select_representative_indices(y_test, result.scores, label=1),
    }


def _build_representative_mapping_rows(
    demo: DemoData,
    result: DemoResult,
    selected: dict[str, list[int]],
    estimator_index: int | None = None,
) -> list[MappingRow]:
    rows: list[MappingRow] = []

    for label_name, indices in selected.items():
        for rank, sample_index in enumerate(indices, start=1):
            phi_row = result.phi_dense[sample_index]
            if estimator_index is not None:
                phi_row = _estimator_phi_block(result.model, phi_row, estimator_index)
            phi_indices, phi_values = _format_phi_mapping(phi_row)
            point = demo.X_test[sample_index]
            rows.append(
                MappingRow(
                    variant=result.name,
                    label_name=label_name,
                    example_rank=rank,
                    sample_index=sample_index,
                    tag=f"{'N' if label_name == 'normal' else 'A'}{rank}",
                    x=float(point[0]),
                    y=float(point[1]),
                    score=float(result.scores[sample_index]),
                    active_features=int(
                        np.count_nonzero(result.phi_dense[sample_index])
                    ),
                    phi_active_indices=phi_indices,
                    phi_active_values=phi_values,
                )
            )

    return rows


def save_representative_mappings(demo: DemoData, results: list[DemoResult]):
    os.makedirs(os.path.dirname(MAPPINGS_PATH), exist_ok=True)
    rows: list[MappingRow] = []
    selected = _shared_representative_indices(results[0], demo.y_test)

    for result in results:
        rows.extend(_build_representative_mapping_rows(demo, result, selected=selected))

    pd.DataFrame(asdict(row) for row in rows).to_csv(MAPPINGS_PATH, index=False)


def _mapping_panel_text(mapping_rows: list[MappingRow]) -> str:
    lines = ["Representative mappings"]
    for row in mapping_rows:
        phi_idx = row.phi_active_indices or "-"
        phi_val = row.phi_active_values or "-"
        lines.append(
            f"{row.tag} s{row.sample_index} score={row.score:.3f} phi[{phi_idx}]={phi_val}"
        )
    return "\n".join(lines)


def _plot_single_result(
    fig,
    ax_phi,
    ax_score,
    ax_map,
    demo: DemoData,
    result: DemoResult,
    selected: dict[str, list[int]],
    estimator_index: int,
):
    scores = result.scores
    phi_dense = result.phi_dense
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)

    y_test = demo.y_test
    mapping_rows = _build_representative_mapping_rows(
        demo, result, selected=selected, estimator_index=estimator_index
    )

    order = np.argsort(y_test * 10 + scores)
    phi_sorted = phi_dense[order]
    y_sorted = y_test[order]
    scores_sorted = scores[order]

    heat = ax_phi.imshow(
        phi_sorted, aspect="auto", cmap="magma", interpolation="nearest"
    )
    ax_phi.set_title(f"{result.name} phi(x)   AUC={result.auc:.3f}")
    ax_phi.set_xlabel("feature index")
    ax_phi.set_ylabel("test sample (sorted by label, then score)")
    fig.colorbar(heat, ax=ax_phi, fraction=0.046, pad=0.04, label="phi value")

    split = int(np.sum(y_sorted == 0))
    ax_phi.axhline(split - 0.5, color="white", linewidth=1.2, linestyle="--")
    ax_phi.text(
        phi_sorted.shape[1] - 0.5,
        split - 1.0,
        "normals above / anomalies below",
        ha="right",
        va="bottom",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.25, "pad": 3, "edgecolor": "none"},
    )

    x_axis = np.arange(len(scores_sorted))
    colors = np.where(y_sorted == 1, "#d7301f", "#2c7fb8")
    ax_score.bar(x_axis, scores_sorted, color=colors, width=0.85)
    ax_score.set_title(f"{result.name} IDK-style anomaly score")
    ax_score.set_xlabel("sorted test sample")
    ax_score.set_ylabel("anomaly score")
    ax_score.set_ylim(0.0, 1.05)
    ax_score.grid(axis="y", alpha=0.2)

    ax_map.axis("off")
    ax_map.text(
        0.0,
        1.0,
        _mapping_panel_text(mapping_rows),
        transform=ax_map.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={
            "facecolor": "#f7f7f7",
            "edgecolor": "#d0d0d0",
            "boxstyle": "round,pad=0.4",
        },
    )


def _plot_shared_geometry(
    ax_geo,
    demo: DemoData,
    result: DemoResult,
    selected: dict[str, list[int]],
    estimator_index: int,
):
    X_train = demo.X_train
    X_test = demo.X_test
    y_test = demo.y_test
    model = result.model
    assert model._centroids is not None
    assert model._radius is not None
    centers = model._centroids[estimator_index]
    radii = np.sqrt(model._radius[estimator_index])
    mapping_rows = _build_representative_mapping_rows(demo, result, selected=selected)

    ax_geo.scatter(
        X_train[:, 0],
        X_train[:, 1],
        s=24,
        c="#c7cedb",
        alpha=0.65,
        label="training normals",
    )
    normal_mask = y_test == 0
    anomaly_mask = y_test == 1
    ax_geo.scatter(
        X_test[normal_mask, 0],
        X_test[normal_mask, 1],
        s=60,
        c="#2c7fb8",
        edgecolor="white",
        linewidth=0.7,
        label="test normal",
    )
    ax_geo.scatter(
        X_test[anomaly_mask, 0],
        X_test[anomaly_mask, 1],
        s=85,
        c="#d7301f",
        marker="X",
        edgecolor="white",
        linewidth=0.7,
        label="test anomaly",
    )
    ax_geo.scatter(
        centers[:, 0],
        centers[:, 1],
        s=150,
        c="#f4a261",
        edgecolor="#111111",
        linewidth=1.0,
        label=f"estimator {estimator_index} centers",
        zorder=5,
    )

    cmap = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(centers)))
    for idx, (center, radius, color) in enumerate(zip(centers, radii, cmap)):
        ax_geo.add_patch(
            Circle(
                xy=center,
                radius=float(radius),
                fill=False,
                lw=1.7,
                alpha=0.75,
                edgecolor=color,
            )
        )
        ax_geo.text(
            center[0],
            center[1],
            str(idx),
            fontsize=9,
            fontweight="bold",
            color="#111111",
            ha="center",
            va="center",
            zorder=6,
        )

    for row in mapping_rows:
        sample_index = row.sample_index
        point = X_test[sample_index]
        color = "#0b4f8a" if row.label_name == "normal" else "#8c1d18"
        ax_geo.text(
            float(point[0]) + 0.05,
            float(point[1]) + 0.05,
            row.tag,
            fontsize=9,
            fontweight="bold",
            color=color,
            bbox={"facecolor": "white", "alpha": 0.75, "pad": 1.5, "edgecolor": color},
            zorder=6,
        )

    ax_geo.set_title(f"{result.name} hyperspheres")
    ax_geo.set_xlabel("x1")
    ax_geo.set_ylabel("x2")
    ax_geo.set_aspect("equal", adjustable="box")
    ax_geo.legend(loc="upper left", frameon=True)
    ax_geo.grid(alpha=0.2)


def plot_demo(demo: DemoData, results: list[DemoResult], estimator_index: int = 0):
    os.makedirs(os.path.dirname(FIGURE_PATH), exist_ok=True)
    selected = _shared_representative_indices(results[0], demo.y_test)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = GridSpec(
        3,
        3,
        figure=fig,
        width_ratios=[1.2, 1.0, 1.0],
        height_ratios=[1.0, 0.5, 0.45],
    )

    ax_geo = fig.add_subplot(grid[:, 0])
    axes = [
        (
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[2, 1]),
        ),
        (
            fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[1, 2]),
            fig.add_subplot(grid[2, 2]),
        ),
    ]

    _plot_shared_geometry(
        ax_geo=ax_geo,
        demo=demo,
        result=results[0],
        selected=selected,
        estimator_index=estimator_index,
    )

    for axis_group, result in zip(axes, results, strict=True):
        _plot_single_result(
            fig=fig,
            ax_phi=axis_group[0],
            ax_score=axis_group[1],
            ax_map=axis_group[2],
            demo=demo,
            result=result,
            selected=selected,
            estimator_index=estimator_index,
        )

    fig.suptitle(
        "Synthetic 2D anomaly detection with IK_INNE: non-overlapping vs overlapping",
        fontsize=16,
    )

    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main():
    n_estimators = 100
    max_samples = 16
    random_state = 8
    r = 30.0
    demo = make_demo_data()
    non_overlapping = IK_INNE(
        n_estimators, max_samples, random_state, overlapping=False, r=r
    )
    overlapping = IK_INNE(
        n_estimators,
        max_samples,
        random_state,
        overlapping=True,
        r=r,
    )
    non_overlapping.fit(demo.X_train)
    overlapping.fit(demo.X_train)

    results = [
        build_demo_result(
            name="iNNE",
            model=non_overlapping,
            X_train=demo.X_train,
            X_test=demo.X_test,
            y_test=demo.y_test,
        ),
        build_demo_result(
            name="iNNE overlapping",
            model=overlapping,
            X_train=demo.X_train,
            X_test=demo.X_test,
            y_test=demo.y_test,
        ),
    ]

    save_scores(demo, results)
    save_representative_mappings(demo, results)
    plot_demo(demo, results)

    print(f"Saved figure: {FIGURE_PATH}")
    print(f"Saved scores: {SCORES_PATH}")
    print(f"Saved mappings: {MAPPINGS_PATH}")
    for result in results:
        print(
            f"{result.name}: AUC={result.auc:.4f}  phi shape={result.phi_dense.shape}  nonzero={int(np.count_nonzero(result.phi_dense))}"
        )


if __name__ == "__main__":
    main()
