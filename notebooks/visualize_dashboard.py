"""
notebooks/visualize_dashboard.py — IK Partitioning Visual Explorer
Run: python notebooks/visualize_dashboard.py → http://127.0.0.1:8054

Tabs:
  1. Geometry Lab    — how each partition cuts the space
  2. All 4 Together  — side-by-side comparison on the same dataset
  3. Kernel View     — IK vs IDK similarity matrices
  4. Anomaly Scores  — which points each partition flags as anomalous
  5. Dataset Browser — explore all 41 datasets
  6. Trade-off       — accuracy vs runtime per condition
  7. Winners         — best partition per dataset, summary chart + table
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from scipy.spatial import Voronoi
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback

from data.datasets  import DATASETS
from src.partitions import get_partition, PARTITION_NAMES

# ══════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════
BG, CARD, CARD2 = "#0f0f1e", "#181830", "#1e1e3a"
BORDER, TEXT, MUTED, ACCENT = "#2e2e5a", "#d0d0f0", "#6666a0", "#6af0f7"
GRID = "#1a1a35"

# Partition colours
PC = {
    'anne':             '#f7a55a',
    'inne':             '#5af0f7',
    'inne-overlapping': '#a07af7',
    'iforest':          '#5af7a0',
    'sciforest':        '#f75aab',
}
PC_RGBA = {
    'anne':             'rgba(247,165,90,0.2)',
    'inne':             'rgba(90,240,247,0.2)',
    'inne-overlapping': 'rgba(160,122,247,0.2)',
    'iforest':          'rgba(90,247,160,0.2)',
    'sciforest':        'rgba(247,90,171,0.2)',
}
# Extended names including inne-overlapping
PNAMES = {**PARTITION_NAMES, 'inne-overlapping': 'Hypersphere-OL (iNNE-OL)'}

CPAL = ['#5af0f7','#f7a55a','#5af7a0','#f75aab','#a07af7','#f7e05a','#5af7d0']
COND_NAME = {1:'Spherical',2:'Elongated',3:'Crescent',4:'Nested',
             5:'Density',6:'High-dim',7:'Large'}
COND_COL  = {1:'#4a90d9',2:'#9b7ff7',3:'#f79b6a',4:'#f76aab',
             5:'#6af7a0',6:'#f7e06a',7:'#aaaaaa'}

BL = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=10, family="monospace"),
    margin=dict(l=50, r=20, t=55, b=40),
    legend=dict(bgcolor=CARD2, bordercolor=BORDER, font=dict(color=TEXT, size=9)),
)

N_EST    = 200
SEED     = 42
MODEL_DIR = os.path.join(ROOT, 'results', 'models')
AUC_PATH  = os.path.join(ROOT, 'results', 'anomaly_detection', 'auc_results.csv')
ARI_PATH  = os.path.join(ROOT, 'results', 'clustering', 'ari_results.csv')

ALL_PARTS  = ['anne', 'inne', 'inne-overlapping', 'iforest', 'sciforest']
GEO_PARTS  = ['anne', 'inne', 'inne-overlapping', 'iforest', 'sciforest']


# ══════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _load_model(ds_name, method, task):
    if not os.path.exists(MODEL_DIR): return None
    tag = 'AD' if task == 'AD' else 'CL'
    pkl = os.path.join(MODEL_DIR, f'{ds_name}_{tag}_{method}.pkl')
    return joblib.load(pkl) if os.path.exists(pkl) else None


def _load_results():
    """Return (auc_df, ari_df) or (None, None)."""
    auc = pd.read_csv(AUC_PATH) if os.path.exists(AUC_PATH) else None
    ari = pd.read_csv(ARI_PATH) if os.path.exists(ARI_PATH) else None
    return auc, ari


def _winner_info():
    """Return {ds_name: {'ad': (partition, score), 'cl': (partition, score)}}."""
    auc, ari = _load_results()
    info = {}
    if auc is not None:
        best = auc.loc[auc.groupby('dataset')['auc_mean'].idxmax()]
        for _, r in best.iterrows():
            info.setdefault(r['dataset'], {})['ad'] = (r['partition'], r['auc_mean'])
    if ari is not None:
        best = ari.loc[ari.groupby('dataset')['ari_mean'].idxmax()]
        for _, r in best.iterrows():
            info.setdefault(r['dataset'], {})['cl'] = (r['partition'], r['ari_mean'])
    return info


# ══════════════════════════════════════════════════════════════════════════
# PROJECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _pca2(X):
    if X.shape[1] == 2: return X, '2D'
    if X.shape[1] == 3: return X[:, :2], 'first 2 of 3 features'
    return PCA(2, random_state=SEED).fit_transform(X), f'PCA 2D (from {X.shape[1]} features)'


def _proj_opts(ds_name):
    ds = DATASETS.get(ds_name)
    if ds is None: return [{'label': 'PCA 2D', 'value': 'pca'}], 'pca'
    f    = ds['features']
    opts = [{'label': 'PCA 2D', 'value': 'pca'}]
    if f >= 2:
        cap = min(f, 8)
        for i in range(cap):
            for j in range(i+1, cap):
                opts.append({'label': f'Feature {i} vs {j}', 'value': f'feat_{i}_{j}'})
    return opts, 'pca'


def _project(X, mode='pca'):
    if mode and mode.startswith('feat_'):
        parts = mode.split('_')
        i, j = int(parts[1]), int(parts[2])
        if X.shape[1] > max(i, j):
            return X[:, [i, j]], f'Feature {i} vs {j}'
    return _pca2(X)


# ══════════════════════════════════════════════════════════════════════════
# GEOMETRY DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _scatter_traces(X2, y, task, showlegend=True):
    out = []
    if task == 'AD':
        for cls in np.unique(y):
            m = y == cls
            out.append(go.Scatter(
                x=X2[m,0].tolist(), y=X2[m,1].tolist(), mode='markers',
                name='anomaly' if cls==1 else 'normal',
                marker=dict(color='#ff5a5a' if cls==1 else '#5af0f7',
                            size=9 if cls==1 else 6,
                            symbol='x' if cls==1 else 'circle', opacity=0.85,
                            line=dict(color='rgba(255,255,255,0.2)', width=0.5)),
                showlegend=showlegend))
    else:
        for i, cls in enumerate(np.unique(y)):
            m = y == cls
            out.append(go.Scatter(
                x=X2[m,0].tolist(), y=X2[m,1].tolist(), mode='markers',
                name=f'class {int(cls)}',
                marker=dict(color=CPAL[i % len(CPAL)], size=6, opacity=0.85,
                            line=dict(color='rgba(255,255,255,0.2)', width=0.5)),
                showlegend=showlegend))
    return out


def _circle_trace(cx, cy, r, color):
    t = np.linspace(0, 2*np.pi, 64)
    return go.Scatter(
        x=(cx + r*np.cos(t)).tolist(), y=(cy + r*np.sin(t)).tolist(),
        mode='lines', line=dict(color=color, width=2), opacity=0.75,
        fill='toself',
        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)',
        showlegend=False, hoverinfo='skip')


def _log_scale_radii(radii, r_min=0.025, r_max=0.11):
    log_r = np.log1p(radii * 1000)
    if log_r.max() > log_r.min():
        return r_min + (r_max-r_min)*(log_r-log_r.min())/(log_r.max()-log_r.min())
    return np.full_like(radii, (r_min+r_max)/2)


def _voronoi_traces(cents, color, xr, yr):
    if len(cents) < 3: return []
    pad = max(xr[1]-xr[0], yr[1]-yr[0]) * 5
    mirrors = np.vstack([cents+[pad,0], cents-[pad,0], cents+[0,pad], cents-[0,pad]])
    try: vor = Voronoi(np.vstack([cents, mirrors]))
    except: return []
    xl, xh, yl, yh = xr[0]-.02, xr[1]+.02, yr[0]-.02, yr[1]+.02
    out = []
    for rp in vor.ridge_vertices:
        if -1 in rp: continue
        v0, v1 = vor.vertices[rp[0]], vor.vertices[rp[1]]
        if not (xl<=v0[0]<=xh and xl<=v1[0]<=xh and yl<=v0[1]<=yh and yl<=v1[1]<=yh): continue
        out.append(go.Scatter(
            x=[float(v0[0]), float(v1[0])], y=[float(v0[1]), float(v1[1])],
            mode='lines', line=dict(color=color, width=1.5),
            opacity=0.6, showlegend=False, hoverinfo='skip'))
    return out


def _iforest_traces(tree, node, x0, x1, y0, y1, color, depth=0):
    if depth > 12 or tree.feature[node] == -2: return []
    f, thr = tree.feature[node], float(tree.threshold[node])
    out = []
    if f == 0:
        out.append(go.Scatter(x=[thr,thr], y=[y0,y1], mode='lines',
            line=dict(color=color, width=1.0), opacity=0.55, showlegend=False, hoverinfo='skip'))
        out += _iforest_traces(tree, tree.children_left[node],  x0, thr, y0, y1, color, depth+1)
        out += _iforest_traces(tree, tree.children_right[node], thr, x1, y0, y1, color, depth+1)
    else:
        out.append(go.Scatter(x=[x0,x1], y=[thr,thr], mode='lines',
            line=dict(color=color, width=1.0), opacity=0.55, showlegend=False, hoverinfo='skip'))
        out += _iforest_traces(tree, tree.children_left[node],  x0, x1, y0, thr, color, depth+1)
        out += _iforest_traces(tree, tree.children_right[node], x0, x1, thr, y1, color, depth+1)
    return out


def _sci_traces(node, x0, x1, y0, y1, color, depth=0):
    if depth > 12 or node.get('leaf'): return []
    feat, coef, split = node['feat'], node['coef'], node['split']
    pts = []
    if len(feat) >= 2:
        c0, c1 = coef[0], coef[1]
        for xv in [x0, x1]:
            if abs(c1) > 1e-9:
                yv = (split - c0*xv) / c1
                if y0 <= yv <= y1: pts.append((xv, yv))
        for yv in [y0, y1]:
            if abs(c0) > 1e-9:
                xv = (split - c1*yv) / c0
                if x0 <= xv <= x1: pts.append((xv, yv))
    elif len(feat) == 1:
        v = split/coef[0] if abs(coef[0]) > 1e-9 else 0
        pts = [(v, y0), (v, y1)] if feat[0] == 0 else [(x0, v), (x1, v)]
    out = []
    if len(pts) >= 2:
        out.append(go.Scatter(
            x=[pts[0][0], pts[1][0]], y=[pts[0][1], pts[1][1]],
            mode='lines', line=dict(color=color, width=1.2),
            opacity=0.6, showlegend=False, hoverinfo='skip'))
    out += _sci_traces(node['left'],  x0, x1, y0, y1, color, depth+1)
    out += _sci_traces(node['right'], x0, x1, y0, y1, color, depth+1)
    return out


def _build_geo_traces(X2, ds, method, part, tree_idx=0):
    """Build all Plotly traces for one partition on 2D data."""
    y     = ds['y']
    color = PC[method]
    pad   = 0.06
    xr    = [float(X2[:,0].min()-pad), float(X2[:,0].max()+pad)]
    yr    = [float(X2[:,1].min()-pad), float(X2[:,1].max()+pad)]
    out   = []

    if method == 'anne':
        model = part._model
        nc    = model.max_samples_
        try:
            seeds    = model._seeds
            cent_idx = seeds[tree_idx*nc:(tree_idx+1)*nc] if np.array(seeds).ndim > 0 else None
        except: cent_idx = None
        if cent_idx is not None and hasattr(cent_idx, '__len__') and len(cent_idx) > 0:
            cents = X2[np.array(cent_idx, dtype=int) % len(X2)]
        else:
            raw   = model.center_data[tree_idx*nc:(tree_idx+1)*nc]
            cents = raw[:, :2] if raw.shape[1] >= 2 else raw
        out += _voronoi_traces(cents, color, xr, yr)
        out.append(go.Scatter(x=cents[:,0].tolist(), y=cents[:,1].tolist(),
            mode='markers', name='centroids',
            marker=dict(color=color, symbol='x', size=12, line=dict(color='white', width=2)),
            showlegend=True))

    elif method in ('inne', 'inne-overlapping'):
        model  = part._model
        c_orig = model._centroids[tree_idx]
        radii  = model._radius[tree_idx]
        c2d    = PCA(2, random_state=SEED).fit_transform(c_orig) if c_orig.shape[1] > 2 else c_orig[:, :2]
        r_vis  = _log_scale_radii(radii)
        for idx in np.argsort(r_vis)[::-1]:
            out.append(_circle_trace(float(c2d[idx,0]), float(c2d[idx,1]), float(r_vis[idx]), color))
        out.append(go.Scatter(x=c2d[:,0].tolist(), y=c2d[:,1].tolist(),
            mode='markers', name='centroids  (big circle = sparse = anomaly)',
            marker=dict(color=color, symbol='cross', size=10, line=dict(color='white', width=2)),
            showlegend=True))

    elif method == 'iforest':
        if ds['features'] <= 2:
            tree = part._model.estimators_[tree_idx].tree_
            out += _iforest_traces(tree, 0, xr[0], xr[1], yr[0], yr[1], color)
        else:
            out.append(go.Scatter(x=[None], y=[None], mode='lines',
                name='axis-parallel splits  (H/V lines only — fits in original space)',
                line=dict(color=color, width=2, dash='dot'), showlegend=True))
        out.append(go.Scatter(x=[None], y=[None], mode='lines',
            name='axis-parallel cuts',
            line=dict(color=color, width=2), showlegend=(ds['features'] <= 2)))

    elif method == 'sciforest':
        if ds['features'] <= 2:
            tree = part._trees[tree_idx]._tree
            out += _sci_traces(tree, xr[0], xr[1], yr[0], yr[1], color)
        out.append(go.Scatter(x=[None], y=[None], mode='lines',
            name='oblique cuts  (any angle)',
            line=dict(color=color, width=2), showlegend=True))

    for t in _scatter_traces(X2, y, ds['task']):
        out.append(t)
    return out, xr, yr


# ══════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def fig_geometry(ds_name, method, psi, tree_idx, proj_mode='pca'):
    """Single partition geometry view."""
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X        = ds['X'].astype(np.float32)
    X2, proj = _project(X, proj_mode)
    part     = _load_model(ds_name, method, ds['task'])
    if part is None:
        part = get_partition(method, n_estimators=N_EST, max_samples=psi, random_state=SEED)
        part.fit(X)
    traces, xr, yr = _build_geo_traces(X2, ds, method, part, tree_idx)
    note = '  ·  circle sizes log-scaled for visibility' if method == 'inne' else ''
    fig  = go.Figure(traces)
    fig.update_layout(**BL, height=530,
        title=f'<b>{PNAMES[method]}</b>  ·  {ds_name}  ·  '
              f'{"AD" if ds["task"]=="AD" else "Clustering"}  ·  '
              f'n={ds["n"]}  ·  ψ={psi}  ·  tree #{tree_idx}  ·  {proj}{note}',
        xaxis=dict(gridcolor=GRID, zeroline=False, range=xr,
                   title='Component 1', scaleanchor='y', scaleratio=1),
        yaxis=dict(gridcolor=GRID, zeroline=False, range=yr, title='Component 2'))
    return fig


def fig_all4(ds_name, psi, proj_mode='pca'):
    """2×3 grid of all 5 partitions on the same dataset."""
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X        = ds['X'].astype(np.float32)
    X2, proj = _project(X, proj_mode)
    # 5 partitions in a 2-row layout: row1=3, row2=2 (centred)
    fig = make_subplots(2, 3,
        subplot_titles=[PNAMES[m] for m in GEO_PARTS] + [''],
        horizontal_spacing=0.04, vertical_spacing=0.10)
    # (row, col, xaxis_suffix, yaxis_suffix)
    ax_map = [(1,1,'',''), (1,2,'2','2'), (1,3,'3','3'),
              (2,1,'4','4'), (2,2,'5','5')]
    for i, method in enumerate(GEO_PARTS):
        row, col, xs, ys = ax_map[i]
        part = _load_model(ds_name, method, ds['task'])
        if part is None:
            part = get_partition(method, n_estimators=N_EST, max_samples=psi, random_state=SEED)
            part.fit(X)
        traces, xr, yr = _build_geo_traces(X2, ds, method, part, 0)
        for t in traces:
            t.showlegend = False
            fig.add_trace(t, row=row, col=col)
        xk   = f'xaxis{xs}' if xs else 'xaxis'
        yk   = f'yaxis{ys}' if ys else 'yaxis'
        xref = f'x{xs}'    if xs else 'x'
        fig.layout[xk].update(range=xr, gridcolor=GRID, zeroline=False, showticklabels=False)
        fig.layout[yk].update(range=yr, gridcolor=GRID, zeroline=False, showticklabels=False,
                               scaleanchor=xref, scaleratio=1)
    # hide the 6th (empty) subplot
    fig.layout['xaxis6'].update(visible=False)
    fig.layout['yaxis6'].update(visible=False)
    fig.update_layout(**BL, height=760,
        title=f'<b>All 5 Partitions</b>  ·  {ds_name}  ·  '
              f'{"AD" if ds["task"]=="AD" else "Clustering"}  ·  '
              f'n={ds["n"]}  ·  ψ={psi}  ·  {proj}')
    return fig


def fig_kernels(ds_name, method, psi):
    """IK and IDK kernel heatmaps side by side, sorted by class."""
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X, y = ds['X'].astype(np.float32), ds['y']
    # subsample for speed
    if len(X) > 400:
        idx = np.random.RandomState(SEED).choice(len(X), 400, replace=False)
        X, y = X[idx], y[idx]
    part = _load_model(ds_name, method, ds['task'])
    if part is None:
        part = get_partition(method, n_estimators=N_EST, max_samples=psi, random_state=SEED)
        part.fit(X)
    order  = np.argsort(y)
    K_ik   = part.similarity_ik(X)[np.ix_(order, order)]
    K_idk  = part.similarity_idk(X)[np.ix_(order, order)]
    bounds = (np.where(np.diff(y[order]) != 0)[0] + 0.5).tolist()
    # Measure how different IK and IDK are
    delta = float(np.abs(K_ik - K_idk).mean())
    delta_note = (f'IK ≈ IDK for this partition  (mean Δ = {delta:.4f})'
                  if delta < 0.01 else
                  f'IK ≠ IDK for this partition  (mean Δ = {delta:.4f}  — normalisation matters here)')

    fig = make_subplots(1, 2,
        subplot_titles=['IK  —  raw co-occurrence', 'IDK  —  normalised'],
        horizontal_spacing=0.12)
    for K, col, cs in [(K_ik, 1, 'Purples'), (K_idk, 2, 'YlOrRd')]:
        fig.add_trace(go.Heatmap(z=K, colorscale=cs, zmin=0, zmax=1, showscale=True,
            colorbar=dict(len=0.85, thickness=12,
                          x=0.44 if col==1 else 1.01, title='similarity')),
            row=1, col=col)
        for b in bounds:
            fig.add_shape(dict(type='line', x0=b, x1=b, y0=-0.5, y1=len(K)-0.5,
                               line=dict(color='white', width=1.0)), row=1, col=col)
            fig.add_shape(dict(type='line', y0=b, y1=b, x0=-0.5, x1=len(K)-0.5,
                               line=dict(color='white', width=1.0)), row=1, col=col)
    fig.update_layout(**BL, height=480,
        title=f'<b>Kernel Matrices</b>  ·  {ds_name}  ·  {PNAMES[method]}  ·  '
              f'n={len(X)} sorted by class')
    fig.update_xaxes(showticklabels=False, gridcolor=GRID, title='samples →')
    fig.update_yaxes(showticklabels=False, gridcolor=GRID, title='samples →')
    return fig, delta_note


def fig_scores(ds_name, psi, kernel='idk', proj_mode='pca'):
    """Anomaly score map for all 5 partitions."""
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X, y     = ds['X'].astype(np.float32), ds['y']
    X2, proj = _project(X, proj_mode)
    auc_df, _ = _load_results()

    fig = make_subplots(1, len(ALL_PARTS),
        subplot_titles=[PNAMES[m] for m in ALL_PARTS],
        horizontal_spacing=0.03)
    idx = np.random.RandomState(SEED).choice(len(X2), min(500, len(X2)), replace=False)

    for mi, method in enumerate(ALL_PARTS):
        col  = mi + 1
        part = _load_model(ds_name, method, ds['task'])
        if part is None:
            part = get_partition(method, n_estimators=N_EST, max_samples=psi, random_state=SEED)
            part.fit(X)
        sc = (1.0 - part.similarity_ik(X).mean(axis=1)) if kernel == 'ik' \
             else part.idk_scores(X)
        cs = 'Blues_r' if kernel == 'ik' else 'RdYlBu_r'

        # Get AUC from results if available
        auc_val = None
        if auc_df is not None and ds['task'] == 'AD':
            rows = auc_df[(auc_df['dataset']==ds_name) &
                          (auc_df['partition']==method) &
                          (auc_df['kernel']==kernel) &
                          (auc_df['max_samples']==psi)]
            if len(rows): auc_val = rows['auc_mean'].iloc[0]

        fig.add_trace(go.Scatter(
            x=X2[idx,0].tolist(), y=X2[idx,1].tolist(), mode='markers',
            marker=dict(color=sc[idx].tolist(), colorscale=cs, cmin=0, cmax=1,
                size=5, opacity=0.85, showscale=(mi == len(ALL_PARTS)-1),
                colorbar=dict(len=0.85, thickness=10,
                    title=dict(text=kernel.upper(), font=dict(color=TEXT, size=10)))
                    if mi == len(ALL_PARTS)-1 else None),
            text=[f'{kernel.upper()}={sc[j]:.3f}  class={int(y[j])}' for j in idx],
            hovertemplate='%{text}<extra></extra>', showlegend=False),
            row=1, col=col)

        if ds['task'] == 'AD':
            ai = np.where(y == 1)[0]
            if len(ai):
                fig.add_trace(go.Scatter(
                    x=X2[ai,0].tolist(), y=X2[ai,1].tolist(), mode='markers',
                    marker=dict(color='rgba(0,0,0,0)', size=11, symbol='circle-open',
                                line=dict(color='#ffff00', width=2)),
                    name='true anomaly' if mi == 0 else '',
                    showlegend=(mi == 0)), row=1, col=col)

        # Add AUC annotation if available
        if auc_val is not None:
            fig.add_annotation(
                text=f'AUC={auc_val:.3f}',
                xref=f'x{col} domain' if col > 1 else 'x domain',
                yref=f'y{col} domain' if col > 1 else 'y domain',
                x=0.5, y=-0.08, showarrow=False,
                font=dict(color=PC.get(method, TEXT), size=10))

    fig.update_xaxes(showticklabels=False, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showticklabels=False, gridcolor=GRID, zeroline=False)
    task_note = ('AD dataset  ·  yellow ring = true anomaly'
                 if ds['task'] == 'AD'
                 else 'Clustering dataset  ·  no anomaly labels')
    fig.update_layout(**BL, height=380,
        title=f'<b>{kernel.upper()} Anomaly Scores</b>  ·  {ds_name}  ·  {task_note}  ·  '
              f'{proj}  ·  red=anomalous  blue=normal')
    return fig


def fig_browser(ds_name, proj_mode='pca'):
    """Dataset scatter + class distribution + winner info."""
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL), []
    X, y     = ds['X'].astype(np.float32), ds['y']
    X2, proj = _project(X, proj_mode)
    winners  = _winner_info().get(ds_name, {})

    fig = make_subplots(1, 2,
        subplot_titles=['Data scatter (true labels)', 'Class / anomaly distribution'],
        column_widths=[0.65, 0.35])
    for t in _scatter_traces(X2, y, ds['task']): fig.add_trace(t, row=1, col=1)
    classes, counts = np.unique(y, return_counts=True)
    task    = ds['task']
    clabels = ['normal' if c==0 else 'anomaly' for c in classes] \
              if task == 'AD' else [f'class {c}' for c in classes]
    ccolors = ['#5af0f7' if c==0 else '#ff5a5a' for c in classes] \
              if task == 'AD' else [CPAL[i % len(CPAL)] for i in range(len(classes))]
    fig.add_trace(go.Bar(x=clabels, y=counts.tolist(), marker_color=ccolors,
        text=[f'{c} ({c/len(y)*100:.0f}%)' for c in counts],
        textposition='outside', showlegend=False, textfont=dict(color=TEXT)),
        row=1, col=2)
    ar = ds.get('anom_rate')
    fig.update_layout(**BL, height=430,
        title=f'<b>{ds_name}</b>  ·  {"AD" if task=="AD" else "Clustering"}  ·  '
              f'C{ds["condition"]} {COND_NAME.get(ds["condition"],"")}  ·  '
              f'n={ds["n"]}  features={ds["features"]}'
              + (f'  ·  anomaly rate {ar:.1f}%' if ar else '')
              + f'<br><sup>{proj}  ·  shape={ds["shape"]}  ·  '
                f'density={ds["density"]}  ·  source={ds["source"]}</sup>')
    fig.update_xaxes(gridcolor=GRID, zeroline=False, row=1, col=1)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, row=1, col=1)
    fig.update_xaxes(gridcolor=GRID, row=1, col=2)
    fig.update_yaxes(gridcolor=GRID, title='count', row=1, col=2)

    # Stats rows
    def _row(label, val, color=TEXT):
        return html.Tr([
            html.Td(label, style={'color':MUTED,'width':'110px','paddingRight':'12px',
                                  'paddingBottom':'4px'}),
            html.Td(val,   style={'color':color,'paddingBottom':'4px'}),
        ])

    ad_winner, cl_winner = winners.get('ad'), winners.get('cl')
    stats = [
        _row('Task',      'Anomaly Detection' if task=='AD' else 'Clustering',
             '#f7a55a' if task=='AD' else '#5af7a0'),
        _row('Condition', f'C{ds["condition"]} — {COND_NAME.get(ds["condition"],"")}',
             COND_COL.get(ds["condition"], TEXT)),
        _row('n',         f'{ds["n"]:,}'),
        _row('Features',  str(ds['features'])),
        _row('Classes',   str(len(np.unique(y)))),
        _row('Shape',     ds['shape']),
        _row('Density',   ds['density']),
        _row('Source',    ds['source']),
    ]
    if ar:
        stats.append(_row('Anomaly rate', f'{ar:.1f}%', '#ff5a5a'))
    if ad_winner:
        p, s = ad_winner
        stats.append(_row('Best AD',  f'{PNAMES.get(p,p)}  AUC={s:.3f}', PC.get(p, ACCENT)))
    if cl_winner:
        p, s = cl_winner
        stats.append(_row('Best CL',  f'{PNAMES.get(p,p)}  ARI={s:.3f}', PC.get(p, ACCENT)))

    return fig, stats


def fig_tradeoff(task_filter='all', cond_filter=0, metric_x='total_time_s'):
    """Accuracy vs runtime — always split AD and Clustering to avoid mixing AUC and ARI."""
    auc_df, ari_df = _load_results()
    if auc_df is None and ari_df is None:
        fig = go.Figure()
        fig.add_annotation(
            text='Run experiments first:<br><br>'
                 'python experiments/run_anomaly.py<br>'
                 'python experiments/run_clustering.py',
            xref='paper', yref='paper', x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color=MUTED), align='center')
        fig.update_layout(**BL, height=350)
        return [fig]

    figs = []

    for df, metric, task_label, score_label in [
        (auc_df, 'auc_mean', 'Anomaly Detection', 'AUC-ROC'),
        (ari_df, 'ari_mean', 'Clustering',        'ARI'),
    ]:
        if df is None: continue
        if task_filter == 'AD'  and task_label != 'Anomaly Detection': continue
        if task_filter == 'C'   and task_label != 'Clustering':        continue
        sub = df[df['condition'] == cond_filter] if cond_filter > 0 else df

        # ── Scatter: accuracy vs runtime ──────────────────────────────────
        fig_scatter = go.Figure()
        for m in ALL_PARTS:
            ms = sub[sub['partition'] == m]
            if len(ms) == 0: continue
            fig_scatter.add_trace(go.Scatter(
                x=ms[metric_x], y=ms[metric],
                mode='markers', name=PNAMES[m],
                marker=dict(color=PC[m], size=9, opacity=0.85,
                            line=dict(color='white', width=0.5)),
                text=[f'<b>{r["dataset"]}</b><br>'
                      f'{score_label}={r[metric]:.3f}<br>'
                      f'time={r[metric_x]:.2f}s<br>'
                      f'C{r["condition"]} {COND_NAME.get(r["condition"],"")}'
                      for _, r in ms.iterrows()],
                hovertemplate='%{text}<extra></extra>'))
        fig_scatter.update_layout(**BL, height=380,
            title=f'<b>{task_label}</b>  ·  {score_label} vs runtime  ·  '
                  f'top-left corner = fast AND accurate  ·  hover for details',
            xaxis=dict(gridcolor=GRID, title=metric_x.replace('_', ' '), type='log'),
            yaxis=dict(gridcolor=GRID, title=score_label, range=[0, 1.05]))
        figs.append(fig_scatter)

        # ── Box: score distribution by condition ──────────────────────────
        fig_box = go.Figure()
        for m in ALL_PARTS:
            ms = sub[sub['partition'] == m]
            if len(ms) == 0: continue
            fig_box.add_trace(go.Box(
                x=[f'C{c} {COND_NAME.get(c,"")}' for c in ms['condition']],
                y=ms[metric], name=PNAMES[m],
                marker_color=PC[m], line_color=PC[m],
                fillcolor=PC_RGBA.get(m, 'rgba(128,128,128,0.2)'),
                boxpoints='all', jitter=0.3, pointpos=-1.8,
                marker=dict(size=5, opacity=0.7)))
        fig_box.update_layout(**BL, height=380,
            title=f'<b>{task_label}</b>  ·  {score_label} by condition  ·  each dot = one dataset',
            xaxis=dict(gridcolor=GRID, title='condition', tickangle=-20),
            yaxis=dict(gridcolor=GRID, title=score_label, range=[0, 1.05]),
            boxmode='group')
        figs.append(fig_box)

        # ── Heatmap: mean score per partition × condition ─────────────────
        pivot = sub.groupby(['partition', 'condition'])[metric].mean().unstack(fill_value=0)
        if len(pivot) > 0:
            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f'C{c} {COND_NAME.get(c,"")}' for c in pivot.columns],
                y=[PNAMES.get(m, m) for m in pivot.index],
                colorscale='RdYlGn', zmin=0, zmax=1,
                text=[[f'{v:.2f}' for v in row] for row in pivot.values],
                texttemplate='%{text}', textfont=dict(size=11, color='white'),
                colorbar=dict(title=score_label, thickness=12)))
            fig_heat.update_layout(**BL, height=220,
                title=f'<b>{task_label}</b>  ·  mean {score_label} per partition × condition  ·  '
                      f'green = good  red = bad',
                xaxis=dict(gridcolor=GRID, tickangle=-20),
                yaxis=dict(gridcolor=GRID))
            figs.append(fig_heat)

    return figs


def _build_winners(task_filter='all', cond_filter=0):
    """Build the full Winners tab content (chart + table)."""
    auc_df, ari_df = _load_results()
    if auc_df is None and ari_df is None:
        return html.P('No results yet. Run experiments first.',
                      style={'color': MUTED, 'padding': '20px'})
    # Apply task filter
    if task_filter == 'AD': ari_df = None
    if task_filter == 'C':  auc_df = None
    if cond_filter > 0:
        if auc_df is not None: auc_df = auc_df[auc_df['condition']==cond_filter]
        if ari_df is not None: ari_df = ari_df[ari_df['condition']==cond_filter]

    # ── Count wins per partition ──────────────────────────────────────────
    ad_wins = {}
    cl_wins = {}
    if auc_df is not None:
        best_ad = auc_df.loc[auc_df.groupby('dataset')['auc_mean'].idxmax()]
        for _, r in best_ad.iterrows():
            ad_wins[r['dataset']] = (r['partition'], r['auc_mean'], r['condition'])
    if ari_df is not None:
        best_cl = ari_df.loc[ari_df.groupby('dataset')['ari_mean'].idxmax()]
        for _, r in best_cl.iterrows():
            cl_wins[r['dataset']] = (r['partition'], r['ari_mean'], r['condition'])

    # ── Bar chart: wins per partition ─────────────────────────────────────
    ad_cnt = {p: sum(1 for v in ad_wins.values() if v[0] == p) for p in ALL_PARTS}
    cl_cnt = {p: sum(1 for v in cl_wins.values() if v[0] == p) for p in ALL_PARTS}

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=[PNAMES.get(p,p) for p in ALL_PARTS],
        y=[ad_cnt[p] for p in ALL_PARTS],
        name='AD wins  (AUC)',
        marker_color=[PC[p] for p in ALL_PARTS], opacity=0.9,
        text=[str(ad_cnt[p]) for p in ALL_PARTS],
        textposition='outside', textfont=dict(color='white')))
    fig_bar.add_trace(go.Bar(
        x=[PNAMES.get(p,p) for p in ALL_PARTS],
        y=[cl_cnt[p] for p in ALL_PARTS],
        name='Clustering wins  (ARI)',
        marker_color=[PC[p] for p in ALL_PARTS], opacity=0.45,
        marker_pattern_shape='/',
        text=[str(cl_cnt[p]) for p in ALL_PARTS],
        textposition='outside', textfont=dict(color='white')))
    fig_bar.update_layout(
        **{**BL, 'margin': dict(l=50, r=20, t=55, b=80)},
        height=300, barmode='group',
        title='<b>Wins per partition</b>  ·  solid = AD (AUC wins)  ·  hatched = Clustering (ARI wins)',
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID, title='# datasets where this partition is best'))

    # ── Table: per dataset ────────────────────────────────────────────────
    th_style = {'color': ACCENT, 'textAlign': 'left', 'padding': '8px 12px',
                'borderBottom': f'2px solid {ACCENT}', 'fontSize': '11px',
                'fontWeight': 'bold', 'letterSpacing': '0.5px'}
    td_style = {'padding': '7px 12px', 'borderBottom': f'1px solid {BORDER}',
                'fontSize': '11px', 'color': TEXT}

    all_ds = sorted(set(list(ad_wins.keys()) + list(cl_wins.keys())))
    rows   = []
    for name in all_ds:
        ad = ad_wins.get(name)
        cl = cl_wins.get(name)
        ds = DATASETS.get(name, {})
        cond = int(ds.get('condition', ad[2] if ad else cl[2]))
        rows.append(html.Tr([
            html.Td(name, style=td_style),
            html.Td(f'C{cond} {COND_NAME.get(cond,"")}',
                    style={**td_style, 'color': COND_COL.get(cond, MUTED)}),
            html.Td([
                html.Span(PNAMES.get(ad[0],ad[0]),
                          style={'color': PC.get(ad[0], TEXT), 'fontWeight': '500'}),
                html.Span(f'  {ad[1]:.3f}', style={'color': MUTED, 'fontSize': '10px'}),
            ] if ad else [html.Span('—', style={'color': MUTED})],
                    style=td_style),
            html.Td([
                html.Span(PNAMES.get(cl[0],cl[0]),
                          style={'color': PC.get(cl[0], TEXT), 'fontWeight': '500'}),
                html.Span(f'  {cl[1]:.3f}', style={'color': MUTED, 'fontSize': '10px'}),
            ] if cl else [html.Span('—', style={'color': MUTED})],
                    style=td_style),
        ]))

    table = html.Table([
        html.Thead(html.Tr([
            html.Th('Dataset',          style=th_style),
            html.Th('Condition',        style=th_style),
            html.Th('Best AD  (AUC)',   style=th_style),
            html.Th('Best CL  (ARI)',   style=th_style),
        ])),
        html.Tbody(rows),
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

    return html.Div([
        dcc.Graph(figure=fig_bar, config={'displayModeBar': False},
                  style={'marginBottom': '16px'}),
        html.P('Best partition per dataset. Click any dataset in Geometry Lab to see why it won.',
               style={'color': MUTED, 'fontSize': '11px', 'marginBottom': '12px'}),
        table,
    ])


# ══════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, suppress_callback_exceptions=True)

TS = dict(backgroundColor='#12122a', color=MUTED, border=f'1px solid {BORDER}',
          borderRadius='8px 8px 0 0', padding='10px 18px',
          fontFamily='monospace', fontSize='12px')
TA = {**TS, 'backgroundColor': CARD, 'color': ACCENT, 'borderBottom': f'2px solid {ACCENT}'}

PSI   = [{'label': f'ψ = {v}', 'value': v} for v in [4, 8, 16, 32, 64]]
TREES = [{'label': f'tree {i}', 'value': i} for i in range(8)]
METHS_GEO = [{'label': PNAMES[m], 'value': m} for m in GEO_PARTS]  # rebuilt from GEO_PARTS
METHS_ALL  = [{'label': PNAMES[m], 'value': m} for m in ALL_PARTS]


def _dd(id_, opts, val, w='100%'):
    return dcc.Dropdown(id=id_, options=opts, value=val, clearable=False,
        style={'width': w, 'backgroundColor': '#12122a', 'color': '#111',
               'border': f'1px solid {BORDER}', 'fontFamily': 'monospace', 'fontSize': '11px'})


def _lbl(t):
    return html.Div(t, style={'color': MUTED, 'fontSize': '10px',
        'letterSpacing': '0.5px', 'marginBottom': '3px', 'marginTop': '10px'})


def _card(*c, mb=10):
    return html.Div(list(c), style={
        'backgroundColor': CARD2, 'borderRadius': '8px',
        'padding': '12px', 'border': f'1px solid {BORDER}', 'marginBottom': f'{mb}px'})


def _head(t, col=ACCENT):
    return html.P(t, style={'color': col, 'fontSize': '10px', 'fontWeight': 'bold',
        'letterSpacing': '1.2px', 'textTransform': 'uppercase', 'margin': '0 0 6px 0'})


def _ds_opts(pool=None, task='all', cond=0, src='all'):
    if pool is None: pool = DATASETS
    def ok(v):
        if task != 'all' and v['task'] != task: return False
        if cond != 0 and v['condition'] != cond: return False
        if src == 'real'  and v['source'] == 'sklearn_gen': return False
        if src == 'synth' and v['source'] != 'sklearn_gen': return False
        return True
    return [{'label': f'[{"REAL" if v["source"]!="sklearn_gen" else "SYN"}] '
                      f'{k}  —  {v["task"]}  C{v["condition"]} {COND_NAME.get(v["condition"],"")}  '
                      f'n={v["n"]}  feat={v["features"]}', 'value': k}
            for k, v in pool.items() if ok(v)]


PDESC = {
    'anne':
        'Voronoi cells: each of the ψ random centroids owns all data closer to it than '
        'to any other centroid. Cells are naturally large in sparse regions and small in dense ones.',
    'inne':
        'Hypersphere partition: each centroid gets a ball whose radius = distance to its nearest '
        'neighbour in the subsample. Big ball = isolated point = likely anomaly. '
        'This density-adaptive property is the key theoretical advantage.',
    'inne-overlapping':
        'Variant of iNNE where spheres are allowed to overlap. '
        'IK and IDK can differ significantly for this partition because overlapping '
        'membership inflates raw co-occurrence counts — IDK normalisation corrects this.',
    'iforest':
        'Axis-parallel cuts: recursive random H/V splits only. Fast and simple. '
        'Each leaf is a hyper-rectangle. Fails when clusters are diagonal, curved, or nested '
        'because cuts cannot follow the data shape.',
    'sciforest':
        'Oblique hyperplane splits: each cut uses a random linear combination of features, '
        'so lines can go diagonal. More flexible than iForest for rotated or elongated structures.',
}
PWHEN = {
    'anne':
        'Best when clusters are compact and well-separated. '
        'Robust baseline across all 7 conditions. Wins most clustering tasks in C1 and C3.',
    'inne':
        'Best for varying-density data (C5) and crescent/irregular shapes (C3). '
        'The density-adaptive balls detect isolated points naturally. '
        'WARNING: struggles on spherical dense data (C1) — radii become too small.',
    'inne-overlapping':
        'Similar strengths to iNNE. The overlapping property can help in nested structures (C4). '
        'Use IDK kernel (not IK) — the normalisation matters significantly for this variant.',
    'iforest':
        'Best for axis-aligned data and large-scale problems (C7). '
        'Wins most AD tasks in C1 and C5. '
        'Avoid on diagonal clusters (C2), curves (C3), or nested shapes (C4).',
    'sciforest':
        'Best for elongated/diagonal clusters (C2) and high-dimensional data (C6). '
        'More flexible than iForest when data has rotational structure. '
        'NOTE: pure-numpy implementation is ~20× slower than iForest.',
}

app.layout = html.Div(
    style={'backgroundColor': BG, 'minHeight': '100vh',
           'fontFamily': 'monospace', 'padding': '20px 24px'},
    children=[
        html.Div([
            html.H1('IK Partitioning — Visual Explorer',
                    style={'color': 'white', 'margin': '0',
                           'fontSize': '18px', 'letterSpacing': '3px'}),
            html.P('Geometry Lab  ·  All 5 Together  ·  Kernel View  ·  Anomaly Scores  ·  '
                   'Dataset Browser  ·  Trade-off  ·  Winners',
                   style={'color': MUTED, 'margin': '4px 0 0 0', 'fontSize': '10px'}),
        ], style={'marginBottom': '16px', 'borderBottom': f'1px solid {BORDER}',
                  'paddingBottom': '14px'}),

        dcc.Tabs(id='tabs', value='geo', children=[
            dcc.Tab(label='Geometry Lab',       value='geo',      style=TS, selected_style=TA),
            dcc.Tab(label='All 5 Together',     value='all4',     style=TS, selected_style=TA),
            dcc.Tab(label='Kernel View',        value='kernels',  style=TS, selected_style=TA),
            dcc.Tab(label='Anomaly Scores',     value='scores',   style=TS, selected_style=TA),
            dcc.Tab(label='Dataset Browser',    value='browser',  style=TS, selected_style=TA),
            dcc.Tab(label='Trade-off',          value='tradeoff', style=TS, selected_style=TA),
            dcc.Tab(label='Winners',            value='winners',  style=TS, selected_style=TA),
        ]),
        html.Div(id='tab-content',
                 style={'backgroundColor': CARD, 'borderRadius': '0 10px 10px 10px',
                        'border': f'1px solid {BORDER}', 'padding': '20px',
                        'minHeight': '600px'}),
    ])


@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render(tab):

    # ── 1. Geometry Lab ───────────────────────────────────────────────────
    if tab == 'geo':
        opts = _ds_opts()
        return html.Div([
            html.Div([
                # LEFT: fixed control panel
                html.Div([
                    _head('Dataset'),
                    _dd('g-ds', opts, list(DATASETS.keys())[0]),

                    _lbl('Partition'),
                    _dd('g-m', METHS_GEO, 'inne'),

                    _lbl('ψ  (subsample size per tree)'),
                    _dd('g-ps', PSI, 16),

                    _lbl('Tree  (one estimator of the ensemble)'),
                    _dd('g-tr', TREES, 0),

                    _lbl('Projection  (for high-dim datasets)'),
                    _dd('g-proj', [], 'pca'),

                    html.Div(style={'height': '14px'}),
                    _card(_head('How ψ works', '#f7a55a'),
                          html.P('Small ψ → few large cells → coarse partition.\n'
                                 'Large ψ → many small cells → fine partition.\n'
                                 'Experiments used ψ=16.',
                                 style={'color': TEXT, 'fontSize': '10px',
                                        'lineHeight': '1.7', 'margin': '0',
                                        'whiteSpace': 'pre-line'}), mb=8),
                    _card(_head('This partition', '#5af7a0'),
                          html.Div(id='g-desc',
                                   style={'color': TEXT, 'fontSize': '10px',
                                          'lineHeight': '1.7'}), mb=8),
                    _card(_head('When it wins', '#f75aab'),
                          html.Div(id='g-when',
                                   style={'color': TEXT, 'fontSize': '10px',
                                          'lineHeight': '1.7'}), mb=0),
                ], style={'width': '260px', 'flexShrink': '0', 'minWidth': '260px'}),

                # RIGHT: plot
                html.Div([
                    dcc.Graph(id='g-fig', style={'height': '580px'},
                              config={'displayModeBar': True, 'scrollZoom': True}),
                ], style={'flex': '1', 'minWidth': '0', 'overflow': 'hidden'}),
            ], style={'display': 'flex', 'gap': '16px', 'alignItems': 'flex-start'}),
        ])

    # ── 2. All 4 Together ─────────────────────────────────────────────────
    elif tab == 'all4':
        opts = _ds_opts()
        return html.Div([
            html.Div([
                _lbl('Dataset'),
                _dd('a-ds', opts, list(DATASETS.keys())[2]),
            ], style={'maxWidth': '700px', 'marginBottom': '4px'}),
            html.Div([
                html.Div([_lbl('ψ'), _dd('a-ps', PSI, 16, '130px')],
                         style={'marginRight': '16px'}),
                html.Div([_lbl('Projection'), _dd('a-proj', [], 'pca', '220px')]),
            ], style={'display': 'flex', 'marginBottom': '12px'}),
            dcc.Graph(id='a-fig', style={'height': '780px'},
                      config={'displayModeBar': True, 'scrollZoom': True}),
            _card(html.Div([
                html.Span('Voronoi (aNNE): ',  style={'fontWeight':'bold','color':PC['anne']}),
                html.Span('nearest-centroid orange lines.  ', style={'color': TEXT}),
                html.Span('Hypersphere (iNNE): ', style={'fontWeight':'bold','color':PC['inne']}),
                html.Span('cyan circles — big = sparse = likely anomaly (log-scaled).  ',
                          style={'color': TEXT}),
                html.Br(),
                html.Span('Hypersphere-OL (iNNE-OL): ', style={'fontWeight':'bold','color':PC['inne-overlapping']}),
                html.Span('purple circles — overlapping regions, IDK normalisation matters.  ',
                          style={'color': TEXT}),
                html.Br(),
                html.Span('iForest: ', style={'fontWeight':'bold','color':PC['iforest']}),
                html.Span('green H/V lines only — fails on curved/diagonal data.  ',
                          style={'color': TEXT}),
                html.Span('SCiForest: ', style={'fontWeight':'bold','color':PC['sciforest']}),
                html.Span('pink oblique lines at any angle — more flexible.',
                          style={'color': TEXT}),
            ], style={'fontSize': '11px', 'lineHeight': '1.9'}), mb=0),
        ])

    # ── 3. Kernel View ────────────────────────────────────────────────────
    elif tab == 'kernels':
        opts = _ds_opts()
        return html.Div([
            html.Div([_lbl('Dataset'), _dd('k-ds', opts, list(DATASETS.keys())[0])],
                     style={'maxWidth': '700px', 'marginBottom': '4px'}),
            html.Div([_lbl('Partition'), _dd('k-m', METHS_ALL, 'anne', '300px')],
                     style={'marginBottom': '4px'}),
            html.Div([_lbl('ψ'), _dd('k-ps', PSI, 16, '130px')],
                     style={'marginBottom': '12px'}),
            _card(
                _head('How to read this'),
                html.Div([
                    html.P('Samples are sorted by true class. Each cell = similarity between two samples. '
                           'White lines = class boundaries.',
                           style={'color': TEXT, 'fontSize': '11px',
                                  'lineHeight': '1.7', 'margin': '0 0 6px 0'}),
                    html.P([html.Span('Good partition → ', style={'color':'#5af7a0','fontWeight':'bold'}),
                            html.Span('bright diagonal blocks (same class = similar). '
                                      'Off-diagonal = dark.',
                                      style={'color': TEXT})],
                           style={'fontSize': '11px', 'margin': '0 0 6px 0'}),
                    html.P([html.Span('IK (purple, left): ',
                                      style={'color':'#a07af7','fontWeight':'bold'}),
                            html.Span('raw co-occurrence. ',style={'color':TEXT}),
                            html.Span('IDK (orange, right): ',
                                      style={'color':'#f7a55a','fontWeight':'bold'}),
                            html.Span('normalised — more contrast. ',style={'color':TEXT}),
                            html.Span('IK = IDK ',style={'color':'#f7e05a','fontWeight':'bold'}),
                            html.Span('for Voronoi/iForest/SCiForest (Δ=0). '
                                      'Differs only for Hypersphere variants.',
                                      style={'color':MUTED})],
                           style={'fontSize': '11px', 'margin': '0'}),
                ]), mb=10),
            html.Div(id='k-delta-note',
                     style={'color': ACCENT, 'fontSize': '11px',
                            'marginBottom': '8px', 'fontStyle': 'italic'}),
            dcc.Graph(id='k-fig', style={'height': '490px'},
                      config={'displayModeBar': True}),
        ])

    # ── 4. Anomaly Scores ─────────────────────────────────────────────────
    elif tab == 'scores':
        opts = _ds_opts()
        return html.Div([
            html.Div([_lbl('Dataset'), _dd('s-ds', opts, list(DATASETS.keys())[0])],
                     style={'maxWidth': '700px', 'marginBottom': '4px'}),
            html.Div([
                html.Div([_lbl('ψ'), _dd('s-ps', PSI, 16, '130px')],
                         style={'marginRight': '16px'}),
                html.Div([_lbl('Projection'), _dd('s-proj', [], 'pca', '220px')]),
            ], style={'display': 'flex', 'marginBottom': '4px'}),
            html.Div([
                _lbl('Kernel'),
                dcc.RadioItems(id='s-show',
                    options=[
                        {'label': '  IDK  (recommended — normalised)', 'value': 'idk'},
                        {'label': '  IK   (raw)',                       'value': 'ik'},
                    ],
                    value='idk',
                    labelStyle={'display': 'block', 'marginBottom': '3px'},
                    style={'color': TEXT, 'fontSize': '11px', 'marginTop': '4px'}),
                html.P('IK = IDK for Voronoi / iForest / SCiForest. '
                       'Difference only visible for Hypersphere variants.',
                       style={'color': MUTED, 'fontSize': '10px',
                              'marginTop': '5px', 'lineHeight': '1.6'}),
            ], style={'marginBottom': '12px'}),
            _card(html.Div([
                html.Span('Blue = normal.  ', style={'color': '#5af0f7', 'fontWeight': 'bold'}),
                html.Span('Red = anomalous.  ', style={'color': '#ff5a5a', 'fontWeight': 'bold'}),
                html.Span('Yellow ring = true anomaly label.  ',
                          style={'color': '#ffff00', 'fontWeight': 'bold'}),
                html.Span('Scores shown at bottom of each panel when experiments have run. '
                          'Hover any point for exact score.',
                          style={'color': MUTED}),
            ], style={'fontSize': '11px'}), mb=10),
            dcc.Graph(id='s-fig', config={'displayModeBar': True, 'scrollZoom': True}),
        ])

    # ── 5. Dataset Browser ────────────────────────────────────────────────
    elif tab == 'browser':
        return html.Div([
            html.Div([
                # LEFT: filters + stats
                html.Div([
                    _head('Filter'),
                    _lbl('Task'),
                    _dd('b-task', [{'label':'All tasks','value':'all'},
                                   {'label':'Clustering only','value':'C'},
                                   {'label':'Anomaly detection only','value':'AD'}], 'all'),
                    _lbl('Condition'),
                    _dd('b-cond', [{'label':'All conditions','value':0}] +
                        [{'label':f'C{c} — {COND_NAME[c]}','value':c} for c in range(1,8)], 0),
                    _lbl('Source'),
                    _dd('b-src', [{'label':'All (real + synthetic)','value':'all'},
                                  {'label':'Real data only','value':'real'},
                                  {'label':'Synthetic only','value':'synth'}], 'all'),
                    html.Div(id='b-count',
                             style={'color': MUTED, 'fontSize': '10px',
                                    'margin': '8px 0 4px 0'}),
                    _lbl('Select dataset'),
                    _dd('b-ds', _ds_opts(), 'iris'),
                    html.Div(style={'height': '16px'}),
                    _head('Stats  (incl. winner from experiments)'),
                    html.Table(id='b-stats',
                               style={'fontSize': '11px', 'lineHeight': '2.1',
                                      'color': TEXT, 'width': '100%'}),
                ], style={'width': '260px', 'flexShrink': '0', 'minWidth': '260px'}),

                # RIGHT: plot
                html.Div([
                    html.Div([_lbl('Projection'), _dd('b-proj', [], 'pca', '220px')],
                             style={'marginBottom': '8px'}),
                    dcc.Graph(id='b-fig', style={'height': '480px'},
                              config={'displayModeBar': True, 'scrollZoom': True}),
                ], style={'flex': '1', 'minWidth': '0', 'overflow': 'hidden'}),
            ], style={'display': 'flex', 'gap': '16px', 'alignItems': 'flex-start'}),
        ])

    # ── 6. Trade-off ──────────────────────────────────────────────────────
    elif tab == 'tradeoff':
        return html.Div([
            _card(html.P(
                'AD and Clustering are always shown separately — AUC and ARI are different metrics '
                'and cannot be plotted on the same axis. '
                'Note: SCiForest runtime (~830s AD) is much higher than others because of the '
                'pure-numpy implementation. Use log scale on X axis.',
                style={'color': TEXT, 'fontSize': '11px',
                       'lineHeight': '1.7', 'margin': '0'}), mb=12),
            html.Div([
                html.Div([
                    _lbl('Show task'),
                    dcc.RadioItems(id='t-task',
                        options=[
                            {'label': '  Both',              'value': 'all'},
                            {'label': '  Anomaly Detection', 'value': 'AD'},
                            {'label': '  Clustering',        'value': 'C'},
                        ],
                        value='all', inline=True,
                        labelStyle={'marginRight': '16px'},
                        style={'color': TEXT, 'fontSize': '11px', 'marginTop': '6px'}),
                ], style={'marginRight': '24px'}),
                html.Div([
                    _lbl('Filter by condition'),
                    _dd('t-cond', [{'label':'All conditions','value':0}] +
                        [{'label':f'C{c} — {COND_NAME[c]}','value':c} for c in range(1,8)],
                        0, '200px'),
                ], style={'marginRight': '16px'}),
                html.Div([
                    _lbl('X axis'),
                    _dd('t-xmet', [
                        {'label':'Total runtime (s)', 'value':'total_time_s'},
                        {'label':'Fit time (s)',       'value':'fit_time_s'},
                        {'label':'Transform time (s)', 'value':'transform_time_s'},
                    ], 'total_time_s', '200px'),
                ]),
            ], style={'display': 'flex', 'flexWrap': 'wrap',
                      'marginBottom': '14px', 'alignItems': 'flex-end'}),
            html.Div(id='t-figs'),
        ])

    # ── 7. Winners ────────────────────────────────────────────────────────
    elif tab == 'winners':
        return html.Div([
            html.Div([
                html.Div([
                    _lbl('Show task'),
                    dcc.RadioItems(id='w-task',
                        options=[
                            {'label': '  Both tasks',          'value': 'all'},
                            {'label': '  Anomaly Detection',   'value': 'AD'},
                            {'label': '  Clustering',          'value': 'C'},
                        ],
                        value='all', inline=True,
                        labelStyle={'marginRight': '20px'},
                        style={'color': TEXT, 'fontSize': '11px', 'marginTop': '4px'}),
                ], style={'marginRight': '32px'}),
                html.Div([
                    _lbl('Filter by condition'),
                    _dd('w-cond',
                        [{'label': 'All conditions', 'value': 0}] +
                        [{'label': f'C{c} — {COND_NAME[c]}', 'value': c}
                         for c in range(1, 8)],
                        0, '200px'),
                ]),
            ], style={'display': 'flex', 'alignItems': 'flex-end',
                      'flexWrap': 'wrap', 'marginBottom': '14px'}),
            html.Div(id='w-content'),
        ])


# ══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════════════

@callback(Output('g-proj','options'), Output('g-proj','value'), Input('g-ds','value'))
def cb_geo_proj(ds): return _proj_opts(ds)

@callback(Output('g-fig','figure'), Output('g-desc','children'), Output('g-when','children'),
          Input('g-ds','value'), Input('g-m','value'), Input('g-ps','value'),
          Input('g-tr','value'), Input('g-proj','value'))
def cb_geo(ds, m, ps, tr, proj):
    return (fig_geometry(ds, m, ps or 16, tr or 0, proj or 'pca'),
            PDESC.get(m, ''), PWHEN.get(m, ''))

@callback(Output('a-proj','options'), Output('a-proj','value'), Input('a-ds','value'))
def cb_all4_proj(ds): return _proj_opts(ds)

@callback(Output('a-fig','figure'),
          Input('a-ds','value'), Input('a-ps','value'), Input('a-proj','value'))
def cb_all4(ds, ps, proj): return fig_all4(ds, ps or 16, proj or 'pca')

@callback(Output('k-fig','figure'), Output('k-delta-note','children'),
          Input('k-ds','value'), Input('k-m','value'), Input('k-ps','value'))
def cb_ker(ds, m, ps):
    fig, note = fig_kernels(ds, m, ps or 16)
    return fig, note

@callback(Output('s-proj','options'), Output('s-proj','value'), Input('s-ds','value'))
def cb_scores_proj(ds): return _proj_opts(ds)

@callback(Output('s-fig','figure'),
          Input('s-ds','value'), Input('s-ps','value'),
          Input('s-show','value'), Input('s-proj','value'))
def cb_scores(ds, ps, kernel, proj):
    return fig_scores(ds, ps or 16, kernel or 'idk', proj or 'pca')

@callback(Output('b-proj','options'), Output('b-proj','value'), Input('b-ds','value'))
def cb_browser_proj(ds): return _proj_opts(ds)

@callback(Output('b-fig','figure'), Output('b-stats','children'),
          Output('b-ds','options'),  Output('b-count','children'),
          Input('b-ds','value'), Input('b-task','value'),
          Input('b-cond','value'), Input('b-src','value'), Input('b-proj','value'))
def cb_browser(ds_name, task, cond, src, proj):
    filtered = {k:v for k,v in DATASETS.items()
                if (task=='all' or v['task']==task)
                and (cond==0 or v['condition']==cond)
                and (src=='all'
                     or (src=='real'  and v['source']!='sklearn_gen')
                     or (src=='synth' and v['source']=='sklearn_gen'))}
    opts = _ds_opts(filtered)
    if ds_name not in filtered and filtered: ds_name = next(iter(filtered))
    fig, stats = fig_browser(ds_name, proj or 'pca')
    return fig, stats, opts, f'{len(filtered)} datasets match filter'

@callback(Output('t-figs','children'),
          Input('t-task','value'), Input('t-cond','value'), Input('t-xmet','value'))
def cb_tradeoff(task, cond, xmet):
    figs = fig_tradeoff(task or 'all', cond or 0, xmet or 'total_time_s')
    return [dcc.Graph(figure=f, config={'displayModeBar':True},
                      style={'marginBottom':'12px'}) for f in figs]

@callback(Output('w-content','children'),
          Input('tabs','value'), Input('w-task','value'), Input('w-cond','value'))
def cb_winners(tab, task, cond):
    if tab != 'winners': return []
    return _build_winners(task or 'all', cond or 0)


if __name__ == '__main__':
    print('\n  IK Partition Visualizer → http://127.0.0.1:8054\n')
    app.run(debug=False, use_reloader=False, port=8054)