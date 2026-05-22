"""
notebooks/visualize_dashboard.py  —  Interactive Partition Visualizer
=====================================================================
Dash app that fits partitions on-the-fly and shows:
  • 2D geometry overlays (Voronoi, hyperspheres, axis/oblique splits)
  • Kernel heatmaps (IK / IDK)
  • Phi feature maps
  • Anomaly scores
  • Side-by-side comparison of all 4 partitions

Run:
    python notebooks/visualize_dashboard.py
    → http://127.0.0.1:8054

Style matches notebooks/eda.py (dark theme).
"""

import sys, os, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback

from src.partitions import get_partition, PARTITION_NAMES
from data.datasets import DATASETS

# ── Theme (matches eda.py) ─────────────────────────────────────────────────
BG      = "#0d0d1a"
SURFACE = "#12122a"
CARD    = "#1a1a35"
GRID    = "#2a2a50"
BORDER  = "#3a3a65"
TEXT    = "#cccce8"
MUTED   = "#7777aa"

C_BLUE   = "#6af0f7"
C_PURPLE = "#9b7ff7"
C_ORANGE = "#f79b6a"
C_GREEN  = "#6af7a0"
C_YELLOW = "#f7e06a"
C_PINK   = "#f76aab"
C_TEAL   = "#6af7d0"

PARTITION_COLORS = {
    'anne':      C_ORANGE,
    'inne':      C_BLUE,
    'iforest':   C_GREEN,
    'sciforest': C_PINK,
}

_PARTITION_SHORT = {
    'anne':      'Voronoi',
    'inne':      'Hypersphere',
    'iforest':   'Axis-parallel',
    'sciforest': 'Random hyperplane',
}

# In-memory cache for fitted partitions
_CACHE = {}

# ── Helpers ────────────────────────────────────────────────────────────────

def _get_2d_datasets():
    return [d for d in DATASETS.values() if d['features'] == 2]


def _fit(dataset_name, method, n_estimators):
    """Fit partition, cached. Returns (part, X, y)."""
    key = (dataset_name, method, n_estimators)
    if key in _CACHE:
        return _CACHE[key]
    ds = DATASETS[dataset_name]
    X = ds['X'].astype(np.float32)
    y = ds['y']
    part = get_partition(method, n_estimators=n_estimators,
                         max_samples=16, random_state=42)
    part.fit(X)
    _CACHE[key] = (part, X, y)
    return part, X, y


def _scatter_trace(X, y, name_suffix=''):
    """Return list of Plotly scatter traces per class."""
    traces = []
    labels = np.unique(y)
    palette = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_YELLOW, C_PINK, C_TEAL]
    for i, lab in enumerate(labels):
        mask = y == lab
        traces.append(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode='markers',
            name=f'Class {lab} {name_suffix}'.strip(),
            marker=dict(size=6, color=palette[i % len(palette)],
                        line=dict(color='white', width=0.5)),
            opacity=0.85,
        ))
    return traces


def _voronoi_shapes(partition, color):
    """Return Plotly line shapes for Voronoi ridges."""
    model = partition._model
    if not hasattr(model, 'centroids_'):
        return []
    C = model.centroids_[0]
    if C.shape[1] != 2 or len(C) < 3:
        return []
    try:
        vor = Voronoi(C)
    except Exception:
        return []
    shapes = []
    for vpair in vor.ridge_vertices:
        if -1 in vpair:
            continue
        p0, p1 = vor.vertices[vpair]
        shapes.append(dict(type='line', x0=p0[0], y0=p0[1],
                           x1=p1[0], y1=p1[1],
                           line=dict(color=color, width=1)))
    for c in C:
        shapes.append(dict(type='circle', x0=c[0]-0.01, y0=c[1]-0.01,
                           x1=c[0]+0.01, y1=c[1]+0.01,
                           line=dict(color='red', width=1)))
    return shapes


def _hypersphere_shapes(partition, color):
    """Return Plotly circle shapes for hyperspheres."""
    model = partition._model
    if not (hasattr(model, 'centroids_') and hasattr(model, 'radius_')):
        return []
    C = model.centroids_[0]
    R = model.radius_[0]
    if C.shape[1] != 2:
        return []
    shapes = []
    for c, r in zip(C, R):
        shapes.append(dict(type='circle', x0=c[0]-r, y0=c[1]-r,
                           x1=c[0]+r, y1=c[1]+r,
                           line=dict(color=color, width=1)))
        shapes.append(dict(type='circle', x0=c[0]-0.008, y0=c[1]-0.008,
                           x1=c[0]+0.008, y1=c[1]+0.008,
                           line=dict(color='white', width=1)))
    return shapes


def _iforest_shapes(partition, color):
    """Return shapes for first tree's axis-parallel splits."""
    tree = partition._model.estimators_[0].tree_
    shapes = []
    def rec(node, x0, x1, y0, y1):
        if tree.feature[node] == -2:
            return
        feat = tree.feature[node]
        thr = tree.threshold[node]
        if feat == 0:
            shapes.append(dict(type='line', x0=thr, y0=y0, x1=thr, y1=y1,
                               line=dict(color=color, width=1.2)))
            rec(tree.children_left[node], x0, thr, y0, y1)
            rec(tree.children_right[node], thr, x1, y0, y1)
        else:
            shapes.append(dict(type='line', x0=x0, y0=thr, x1=x1, y1=thr,
                               line=dict(color=color, width=1.2)))
            rec(tree.children_left[node], x0, x1, y0, thr)
            rec(tree.children_right[node], x0, x1, thr, y1)
    # Use data bounds; will be refined when added to figure
    rec(0, 0, 1, 0, 1)
    return shapes


def _sciforest_shapes(partition, color):
    """Return shapes for first tree's oblique splits."""
    tree = partition._trees[0]._tree
    shapes = []
    def rec(node, x0, x1, y0, y1):
        if node.get('leaf'):
            return
        feat = node['feat']
        coef = node['coef']
        split = node['split']
        # line intersections with bbox
        pts = []
        if len(feat) >= 2:
            f0, f1 = feat[0], feat[1]
            c0, c1 = coef[0], coef[1]
            # This assumes original features are 0 and 1; for 2D this holds
            for xv in [x0, x1]:
                if abs(c1) > 1e-9:
                    v = (split - c0 * xv) / c1
                    if y0 <= v <= y1:
                        pts.append((xv, v))
            for yv in [y0, y1]:
                if abs(c0) > 1e-9:
                    v = (split - c1 * yv) / c0
                    if x0 <= v <= x1:
                        pts.append((v, yv))
        elif len(feat) == 1:
            if feat[0] == 0 and abs(coef[0]) > 1e-9:
                xv = split / coef[0]
                pts = [(xv, y0), (xv, y1)]
            elif feat[0] == 1 and abs(coef[0]) > 1e-9:
                yv = split / coef[0]
                pts = [(x0, yv), (x1, yv)]
        if len(pts) >= 2:
            shapes.append(dict(type='line', x0=pts[0][0], y0=pts[0][1],
                               x1=pts[1][0], y1=pts[1][1],
                               line=dict(color=color, width=1.2)))
        rec(node['left'], x0, x1, y0, y1)
        rec(node['right'], x0, x1, y0, y1)
    rec(tree, 0, 1, 0, 1)
    return shapes


def _make_geometry_fig(X, y, partition, method):
    """Create a Plotly figure with partition geometry overlay."""
    fig = go.Figure()
    for tr in _scatter_trace(X, y):
        fig.add_trace(tr)

    x_min, x_max = float(X[:, 0].min() - 0.05), float(X[:, 0].max() + 0.05)
    y_min, y_max = float(X[:, 1].min() - 0.05), float(X[:, 1].max() + 0.05)

    color = PARTITION_COLORS.get(method, '#fff')
    if method == 'anne':
        shapes = _voronoi_shapes(partition, color)
    elif method == 'inne':
        shapes = _hypersphere_shapes(partition, color)
    elif method == 'iforest':
        shapes = _iforest_shapes(partition, color)
    else:
        shapes = _sciforest_shapes(partition, color)

    # Update shape coordinates to actual bounds for tree methods
    for s in shapes:
        if method in ('iforest', 'sciforest'):
            if s['x0'] == 0 and s['x1'] == 1:
                s['x0'] = x_min; s['x1'] = x_max
            if s['y0'] == 0 and s['y1'] == 1:
                s['y0'] = y_min; s['y1'] = y_max

    fig.update_layout(
        shapes=shapes,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=11, family='monospace'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID, zeroline=False, range=[x_min, x_max]),
        yaxis=dict(gridcolor=GRID, zeroline=False, range=[y_min, y_max],
                   scaleanchor='x', scaleratio=1),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
        title=dict(text=f'{_PARTITION_SHORT[method]} overlay',
                   font=dict(color=TEXT, size=12)),
        height=500,
    )
    return fig


def _make_kernel_fig(K, y, title):
    order = np.argsort(y)
    Ks = K[np.ix_(order, order)]
    fig = go.Figure(data=go.Heatmap(
        z=Ks, colorscale='Viridis', zmin=0, zmax=1,
        hovertemplate='i=%{x}<br>j=%{y}<br>K=%{z:.3f}<extra></extra>',
    ))
    # class boundary lines
    boundaries = np.where(np.diff(y[order]) != 0)[0]
    for b in boundaries:
        fig.add_vline(x=b + 0.5, line=dict(color='white', width=1))
        fig.add_hline(y=b + 0.5, line=dict(color='white', width=1))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=11, family='monospace'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID, showticklabels=False),
        yaxis=dict(gridcolor=GRID, showticklabels=False, autorange='reversed'),
        title=dict(text=title, font=dict(color=TEXT, size=12)),
        height=480,
    )
    return fig


def _make_phi_fig(phi):
    if hasattr(phi, 'toarray'):
        D = phi.toarray()
    else:
        D = np.asarray(phi)
    # Show at most 200 samples to keep it responsive
    n_show = min(200, D.shape[0])
    fig = go.Figure(data=go.Heatmap(
        z=D[:n_show, :], colorscale=[[0, CARD], [1, C_BLUE]],
        hovertemplate='sample=%{y}<br>cell=%{x}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=11, family='monospace'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID, title='cell index'),
        yaxis=dict(gridcolor=GRID, title='sample index', autorange='reversed'),
        title=dict(text='Φ feature map (binary)', font=dict(color=TEXT, size=12)),
        height=480,
    )
    return fig


def _make_scores_fig(X, y, scores):
    fig = go.Figure(data=go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers',
        marker=dict(size=7, color=scores, colorscale='RdYlBu_r',
                    cmin=0, cmax=1, line=dict(color='white', width=0.5)),
        text=[f'score={s:.3f}' for s in scores],
        hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>%{text}<extra></extra>',
    ))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=11, family='monospace'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False, scaleanchor='x', scaleratio=1),
        title=dict(text='IDK anomaly scores (red = anomalous)',
                   font=dict(color=TEXT, size=12)),
        height=500,
    )
    return fig


def _make_compare_fig(X, y, partitions):
    """2×2 comparison figure."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[_PARTITION_SHORT[m] for m in partitions.keys()],
                        horizontal_spacing=0.08, vertical_spacing=0.12)
    x_min, x_max = float(X[:, 0].min() - 0.05), float(X[:, 0].max() + 0.05)
    y_min, y_max = float(X[:, 1].min() - 0.05), float(X[:, 1].max() + 0.05)

    for idx, (method, part) in enumerate(partitions.items()):
        row = idx // 2 + 1
        col = idx % 2 + 1
        color = PARTITION_COLORS[method]

        # Data points
        for lab in np.unique(y):
            mask = y == lab
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode='markers',
                name=f'Class {lab}',
                marker=dict(size=5, line=dict(color='white', width=0.3)),
                showlegend=(idx == 0),
                legendgroup=f'cls{lab}',
            ), row=row, col=col)

        # Shapes
        if method == 'anne':
            shapes = _voronoi_shapes(part, color)
        elif method == 'inne':
            shapes = _hypersphere_shapes(part, color)
        elif method == 'iforest':
            shapes = _iforest_shapes(part, color)
        else:
            shapes = _sciforest_shapes(part, color)

        for s in shapes:
            if method in ('iforest', 'sciforest'):
                if s.get('x0') == 0 and s.get('x1') == 1:
                    s['x0'] = x_min; s['x1'] = x_max
                if s.get('y0') == 0 and s.get('y1') == 1:
                    s['y0'] = y_min; s['y1'] = y_max
            # add_shape with subplot references
            xref = f'x{idx+1}' if idx > 0 else 'x'
            yref = f'y{idx+1}' if idx > 0 else 'y'
            fig.add_shape({**s, 'xref': xref, 'yref': yref})

        fig.update_xaxes(gridcolor=GRID, zeroline=False, range=[x_min, x_max],
                         row=row, col=col)
        fig.update_yaxes(gridcolor=GRID, zeroline=False, range=[y_min, y_max],
                         scaleanchor=f'x{idx+1}' if idx > 0 else 'x',
                         scaleratio=1, row=row, col=col)

    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=10, family='monospace'),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
        title=dict(text='Partition comparison', font=dict(color=TEXT, size=14)),
        height=800,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════

def card(children, mb=16):
    return html.Div(children, style={
        'backgroundColor': CARD, 'borderRadius': '12px',
        'padding': '18px', 'border': f'1px solid {BORDER}',
        'marginBottom': f'{mb}px'
    })


def sec(text, color=C_PURPLE):
    return html.H3(text, style={
        'color': color, 'marginBottom': '12px', 'marginTop': '0',
        'fontSize': '12px', 'letterSpacing': '1.5px',
        'textTransform': 'uppercase',
    })


# ══════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════

def run():
    ds_2d = _get_2d_datasets()
    if not ds_2d:
        print('No 2D datasets found.')
        return

    ds_options = [{'label': d['name'], 'value': d['name']} for d in ds_2d]
    part_options = [
        {'label': PARTITION_NAMES[m], 'value': m}
        for m in ['anne', 'inne', 'iforest', 'sciforest']
    ]
    part_options.insert(0, {'label': 'All 4 (compare)', 'value': 'all'})

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    tab_s = {"backgroundColor": SURFACE, "color": MUTED,
             "border": f"1px solid {BORDER}", "borderRadius": "8px 8px 0 0",
             "padding": "10px 20px", "fontFamily": "monospace", "fontSize": "13px"}
    tab_a = {**tab_s, "backgroundColor": CARD, "color": C_BLUE,
             "borderBottom": f"2px solid {C_BLUE}"}

    app.layout = html.Div(
        style={"backgroundColor": BG, "minHeight": "100vh",
               "fontFamily": "monospace", "padding": "24px"},
        children=[
            # Header
            html.Div([
                html.H1("Partition Visualizer",
                        style={"color": "white", "margin": "0",
                               "fontSize": "20px", "letterSpacing": "3px"}),
                html.P("Interactive geometry, kernels, and scores for 2D datasets",
                       style={"color": MUTED, "margin": "6px 0 0 0", "fontSize": "11px"}),
            ], style={"marginBottom": "20px",
                      "borderBottom": f"1px solid {BORDER}", "paddingBottom": "16px"}),

            # Controls
            card([
                html.Div([
                    html.Div([
                        html.P("Dataset", style={"color": MUTED, "fontSize": "10px", "margin": "0 0 4px 0"}),
                        dcc.Dropdown(id='vd-ds', options=ds_options,
                                     value=ds_2d[0]['name'],
                                     style={"width": "260px", "backgroundColor": SURFACE,
                                            "color": "#111", "border": f"1px solid {BORDER}"}),
                    ], style={"marginRight": "16px"}),
                    html.Div([
                        html.P("Partition", style={"color": MUTED, "fontSize": "10px", "margin": "0 0 4px 0"}),
                        dcc.Dropdown(id='vd-part', options=part_options, value='anne',
                                     style={"width": "240px", "backgroundColor": SURFACE,
                                            "color": "#111", "border": f"1px solid {BORDER}"}),
                    ], style={"marginRight": "16px"}),
                    html.Div([
                        html.P("n_estimators", style={"color": MUTED, "fontSize": "10px", "margin": "0 0 4px 0"}),
                        dcc.Slider(id='vd-nest', min=25, max=200, step=25, value=50,
                                   marks={25: '25', 50: '50', 100: '100', 200: '200'},
                                   tooltip={"placement": "bottom", "always_visible": False}),
                    ], style={"width": "280px", "marginRight": "16px"}),
                    html.Div([
                        html.Button("Fit & Plot", id='vd-go', n_clicks=0,
                                    style={"backgroundColor": C_BLUE, "color": BG,
                                           "border": "none", "borderRadius": "8px",
                                           "padding": "10px 24px", "fontWeight": "bold",
                                           "fontFamily": "monospace", "cursor": "pointer"}),
                    ], style={"display": "flex", "alignItems": "flex-end"}),
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "alignItems": "flex-end"}),
            ], mb=12),

            # Status
            html.Div(id='vd-status', style={"color": C_GREEN, "fontSize": "12px", "marginBottom": "12px"}),

            # Tabs
            dcc.Tabs(id='vd-tabs', value='geometry', children=[
                dcc.Tab(label="Geometry",  value='geometry',  style=tab_s, selected_style=tab_a),
                dcc.Tab(label="IK Kernel", value='ik',        style=tab_s, selected_style=tab_a),
                dcc.Tab(label="IDK Kernel",value='idk',       style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Phi Map",   value='phi',       style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Scores",    value='scores',    style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Compare",   value='compare',   style=tab_s, selected_style=tab_a),
            ]),

            html.Div(id='vd-content',
                     style={"backgroundColor": CARD, "borderRadius": "0 12px 12px 12px",
                            "border": f"1px solid {BORDER}", "padding": "18px",
                            "minHeight": "540px"}),
        ]
    )

    @callback(
        Output('vd-content', 'children'),
        Output('vd-status', 'children'),
        Input('vd-go', 'n_clicks'),
        Input('vd-tabs', 'value'),
        State('vd-ds', 'value'),
        State('vd-part', 'value'),
        State('vd-nest', 'value'),
    )
    def update(n_clicks, tab, ds_name, method, n_est):
        if n_clicks == 0:
            return html.P("Click 'Fit & Plot' to generate visualisations.",
                          style={"color": MUTED}), ""

        ds = DATASETS[ds_name]
        X = ds['X'].astype(np.float32)
        y = ds['y']
        status = f"Fitted on {ds_name}  n={len(X)}  est={n_est}"

        try:
            if method == 'all':
                if tab != 'compare':
                    return html.P("'All 4' mode only available in the Compare tab.",
                                  style={"color": C_ORANGE}), status
                parts = {}
                for m in ['anne', 'inne', 'iforest', 'sciforest']:
                    p, _, _ = _fit(ds_name, m, n_est)
                    parts[m] = p
                fig = _make_compare_fig(X, y, parts)
                return dcc.Graph(figure=fig, style={"height": "820px"}), status

            part, _, _ = _fit(ds_name, method, n_est)

            if tab == 'geometry':
                fig = _make_geometry_fig(X, y, part, method)
                return dcc.Graph(figure=fig, style={"height": "540px"}), status

            if tab == 'ik':
                K = part.similarity_ik(X)
                fig = _make_kernel_fig(K, y, f'IK kernel — {PARTITION_NAMES[method]}')
                return dcc.Graph(figure=fig, style={"height": "540px"}), status

            if tab == 'idk':
                K = part.similarity_idk(X)
                fig = _make_kernel_fig(K, y, f'IDK kernel — {PARTITION_NAMES[method]}')
                return dcc.Graph(figure=fig, style={"height": "540px"}), status

            if tab == 'phi':
                phi = part.transform(X)
                fig = _make_phi_fig(phi)
                return dcc.Graph(figure=fig, style={"height": "540px"}), status

            if tab == 'scores':
                scores = part.idk_scores(X)
                fig = _make_scores_fig(X, y, scores)
                return dcc.Graph(figure=fig, style={"height": "540px"}), status

            if tab == 'compare':
                return html.P("Select a single partition and use the Geometry tab, "
                              "or choose 'All 4' and use this tab.",
                              style={"color": C_ORANGE}), status

        except Exception as e:
            import traceback
            return html.Pre(traceback.format_exc(),
                            style={"color": C_PINK, "fontSize": "11px"}), str(e)

        return html.P("Select a tab.", style={"color": MUTED}), ""

    print(f"\n  Partition Visualizer → http://127.0.0.1:8054\n")
    app.run(debug=False, use_reloader=False, port=8054)


if __name__ == '__main__':
    run()
