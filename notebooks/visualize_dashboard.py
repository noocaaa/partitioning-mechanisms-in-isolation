"""
notebooks/visualize_dashboard.py — IK Partitioning Visual Explorer
Run: python notebooks/visualize_dashboard.py → http://127.0.0.1:8054
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

# ── Theme ──────────────────────────────────────────────────────────────────
BG, CARD, CARD2 = "#0f0f1e", "#181830", "#1e1e3a"
BORDER, TEXT, MUTED, ACCENT = "#2e2e5a", "#d0d0f0", "#6666a0", "#6af0f7"
GRID = "#1a1a35"
PC   = {'anne':'#f7a55a','inne':'#5af0f7','inne-overlapping':'#a07af7','iforest':'#5af7a0','sciforest':'#f75aab'}
PC_RGBA = {'anne':'rgba(247,165,90,0.2)','inne':'rgba(90,240,247,0.2)',
          'inne-overlapping':'rgba(160,122,247,0.2)',
          'iforest':'rgba(90,247,160,0.2)','sciforest':'rgba(247,90,171,0.2)'}
CPAL = ['#5af0f7','#f7a55a','#5af7a0','#f75aab','#a07af7','#f7e05a','#5af7d0']
COND_NAME = {1:'Spherical',2:'Elongated',3:'Crescent',4:'Nested',5:'Density',6:'High-dim',7:'Large'}
COND_COL  = {1:'#4a90d9',2:'#9b7ff7',3:'#f79b6a',4:'#f76aab',5:'#6af7a0',6:'#f7e06a',7:'#aaaaaa'}

BL = dict(paper_bgcolor=CARD, plot_bgcolor=CARD,
          font=dict(color=TEXT,size=10,family="monospace"),
          margin=dict(l=50,r=20,t=55,b=40),
          legend=dict(bgcolor=CARD2,bordercolor=BORDER,font=dict(color=TEXT,size=9)))

N_EST, SEED = 200, 42
MODEL_DIR = os.path.join(ROOT, 'results', 'models')

# All datasets can be shown in geometry (project to 2D if needed)
DS_GEO = DATASETS   # all datasets, we PCA if >2D

def _load_model(ds_name, method, task):
    """Load a saved winner model if available, else return None."""
    if not os.path.exists(MODEL_DIR):
        return None
    tag = 'AD' if task == 'AD' else 'CL'
    pkl = os.path.join(MODEL_DIR, f'{ds_name}_{tag}_{method}.pkl')
    if os.path.exists(pkl):
        return joblib.load(pkl)
    return None

def _pca2(X):
    if X.shape[1] == 2: return X, '2D original'
    if X.shape[1] == 3: return X[:,:2], '2D (first 2 dims)'
    X2 = PCA(2, random_state=SEED).fit_transform(X)
    return X2, f'PCA 2D (from {X.shape[1]} feat)'


def _proj_opts(ds_name):
    """Generate projection-mode dropdown options for a dataset."""
    ds = DATASETS.get(ds_name)
    if ds is None:
        return [{'label': 'PCA 2D', 'value': 'pca'}], 'pca'
    f = ds['features']
    opts = [{'label': 'PCA 2D', 'value': 'pca'}]
    if f == 2:
        opts.append({'label': 'Feature 0 vs 1 (original)', 'value': 'feat_0_1'})
    elif f > 2:
        max_f = min(f, 12)
        for i in range(max_f):
            for j in range(i + 1, max_f):
                opts.append({'label': f'Feat {i} vs {j}', 'value': f'feat_{i}_{j}'})
    return opts, 'pca'


def _project2(X, mode='pca'):
    """Project data to 2D using PCA or a specific feature pair."""
    if mode and mode.startswith('feat_'):
        parts = mode.split('_')
        i, j = int(parts[1]), int(parts[2])
        if X.shape[1] > max(i, j):
            return X[:, [i, j]], f'Feature {i} vs {j} (from {X.shape[1]}D)'
    return _pca2(X)


# ══════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _scatter(X2, y, task, size=6, showlegend=True):
    out = []
    if task == 'AD':
        for cls in np.unique(y):
            m = y==cls
            out.append(go.Scatter(
                x=X2[m,0].tolist(), y=X2[m,1].tolist(), mode='markers',
                name='anomaly' if cls==1 else 'normal',
                marker=dict(color='#ff5a5a' if cls==1 else '#5af0f7',
                            size=size+3 if cls==1 else size, symbol='x' if cls==1 else 'circle',
                            opacity=0.85, line=dict(color='rgba(255,255,255,0.2)',width=0.5)),
                showlegend=showlegend))
    else:
        for i,cls in enumerate(np.unique(y)):
            m = y==cls
            out.append(go.Scatter(
                x=X2[m,0].tolist(), y=X2[m,1].tolist(), mode='markers',
                name=f'class {int(cls)}',
                marker=dict(color=CPAL[i%len(CPAL)], size=size, opacity=0.85,
                            line=dict(color='rgba(255,255,255,0.2)',width=0.5)),
                showlegend=showlegend))
    return out


def _circle(cx, cy, r, color):
    t = np.linspace(0, 2*np.pi, 64)
    return go.Scatter(
        x=(cx+r*np.cos(t)).tolist(), y=(cy+r*np.sin(t)).tolist(),
        mode='lines', line=dict(color=color,width=2), opacity=0.75,
        fill='toself',
        fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)',
        showlegend=False, hoverinfo='skip')


def _log_scale_radii(radii, r_min=0.025, r_max=0.11):
    log_r = np.log1p(radii * 1000)
    if log_r.max() > log_r.min():
        return r_min + (r_max-r_min)*(log_r-log_r.min())/(log_r.max()-log_r.min())
    return np.full_like(radii, (r_min+r_max)/2)


def _voronoi_lines(cents, color, xr, yr):
    if len(cents)<3: return []
    pad = max(xr[1]-xr[0], yr[1]-yr[0])*5
    mirrors = np.vstack([cents+[pad,0],cents-[pad,0],cents+[0,pad],cents-[0,pad]])
    try: vor = Voronoi(np.vstack([cents,mirrors]))
    except: return []
    xl,xh,yl,yh = xr[0]-.02,xr[1]+.02,yr[0]-.02,yr[1]+.02
    out = []
    for rp in vor.ridge_vertices:
        if -1 in rp: continue
        v0,v1 = vor.vertices[rp[0]],vor.vertices[rp[1]]
        if not (xl<=v0[0]<=xh and xl<=v1[0]<=xh and yl<=v0[1]<=yh and yl<=v1[1]<=yh): continue
        out.append(go.Scatter(x=[float(v0[0]),float(v1[0])],y=[float(v0[1]),float(v1[1])],
            mode='lines',line=dict(color=color,width=1.5),opacity=0.6,showlegend=False,hoverinfo='skip'))
    return out


def _iforest_lines(tree, node, x0,x1,y0,y1, color, depth=0):
    if depth>12 or tree.feature[node]==-2: return []
    f,thr = tree.feature[node],float(tree.threshold[node])
    out = []
    if f==0:
        out.append(go.Scatter(x=[thr,thr],y=[y0,y1],mode='lines',
            line=dict(color=color,width=1.0),opacity=0.55,showlegend=False,hoverinfo='skip'))
        out+=_iforest_lines(tree,tree.children_left[node], x0,thr,y0,y1,color,depth+1)
        out+=_iforest_lines(tree,tree.children_right[node],thr,x1,y0,y1,color,depth+1)
    else:
        out.append(go.Scatter(x=[x0,x1],y=[thr,thr],mode='lines',
            line=dict(color=color,width=1.0),opacity=0.55,showlegend=False,hoverinfo='skip'))
        out+=_iforest_lines(tree,tree.children_left[node], x0,x1,y0,thr,color,depth+1)
        out+=_iforest_lines(tree,tree.children_right[node],x0,x1,thr,y1,color,depth+1)
    return out


def _sci_lines(node, x0,x1,y0,y1, color, depth=0):
    if depth>12 or node.get('leaf'): return []
    feat,coef,split = node['feat'],node['coef'],node['split']
    pts = []
    if len(feat)>=2:
        c0,c1=coef[0],coef[1]
        for xv in [x0,x1]:
            if abs(c1)>1e-9:
                yv=(split-c0*xv)/c1
                if y0<=yv<=y1: pts.append((xv,yv))
        for yv in [y0,y1]:
            if abs(c0)>1e-9:
                xv=(split-c1*yv)/c0
                if x0<=xv<=x1: pts.append((xv,yv))
    elif len(feat)==1:
        v=split/coef[0] if abs(coef[0])>1e-9 else 0
        pts=[(v,y0),(v,y1)] if feat[0]==0 else [(x0,v),(x1,v)]
    out = []
    if len(pts)>=2:
        out.append(go.Scatter(x=[pts[0][0],pts[1][0]],y=[pts[0][1],pts[1][1]],
            mode='lines',line=dict(color=color,width=1.2),opacity=0.6,showlegend=False,hoverinfo='skip'))
    out+=_sci_lines(node['left'], x0,x1,y0,y1,color,depth+1)
    out+=_sci_lines(node['right'],x0,x1,y0,y1,color,depth+1)
    return out


def geo_traces_2d(X2, ds, method, part, tree_idx=0, include_scatter=True):
    """Build geometry traces on 2D projected data."""
    y = ds['y']
    color = PC[method]
    pad = 0.06
    xr = [float(X2[:,0].min()-pad), float(X2[:,0].max()+pad)]
    yr = [float(X2[:,1].min()-pad), float(X2[:,1].max()+pad)]
    out = []

    if method == 'anne':
        model = part._model
        nc = model.max_samples_
        try:
            seeds = model._seeds
            cent_idx = seeds[tree_idx*nc:(tree_idx+1)*nc] if len(np.array(seeds).shape)>0 else None
        except: cent_idx = None

        if cent_idx is not None and hasattr(cent_idx,'__len__') and len(cent_idx)>0:
            cents_2d = X2[np.array(cent_idx, dtype=int) % len(X2)]
        else:
            cents_orig = model.center_data[tree_idx*nc:(tree_idx+1)*nc]
            cents_2d = np.array(cents_orig)[:,:2] if cents_orig.shape[1]>=2 else cents_orig

        out += _voronoi_lines(cents_2d, color, xr, yr)
        out.append(go.Scatter(x=cents_2d[:,0].tolist(), y=cents_2d[:,1].tolist(),
            mode='markers', name='centroids',
            marker=dict(color=color,symbol='x',size=12,line=dict(color='white',width=2)),
            showlegend=True))

    elif method == 'inne':
        model = part._model
        cents_orig = model._centroids[tree_idx]
        radii = model._radius[tree_idx]
        if cents_orig.shape[1] > 2:
            cents_2d = PCA(2, random_state=SEED).fit_transform(cents_orig)
        else:
            cents_2d = cents_orig[:, :2]
        r_vis = _log_scale_radii(radii)
        order = np.argsort(r_vis)[::-1]
        for idx in order:
            out.append(_circle(float(cents_2d[idx,0]), float(cents_2d[idx,1]),
                               float(r_vis[idx]), color))
        out.append(go.Scatter(x=cents_2d[:,0].tolist(), y=cents_2d[:,1].tolist(),
            mode='markers', name='centroids (+ = centre)',
            marker=dict(color=color,symbol='cross',size=10,line=dict(color='white',width=2)),
            showlegend=True))
        out.append(go.Scatter(x=[None],y=[None],mode='markers',
            name='big circle = sparse region = likely anomaly',
            marker=dict(color=color,symbol='circle-open',size=16,line=dict(color=color,width=2)),
            showlegend=True))

    elif method == 'iforest':
        if ds['features'] <= 2:
            tree = part._model.estimators_[tree_idx].tree_
            out += _iforest_lines(tree, 0, xr[0],xr[1], yr[0],yr[1], color)
        else:
            out.append(go.Scatter(x=[None],y=[None],mode='lines',
                name='splits in original space (PCA projected)',
                line=dict(color=color,width=2,dash='dot'),showlegend=True))
        out.append(go.Scatter(x=[None],y=[None],mode='lines',
            name='axis-parallel cuts (H/V only)',
            line=dict(color=color,width=2),showlegend=True))

    elif method == 'sciforest':
        if ds['features'] <= 2:
            tree = part._trees[tree_idx]._tree
            out += _sci_lines(tree, xr[0],xr[1], yr[0],yr[1], color)
        out.append(go.Scatter(x=[None],y=[None],mode='lines',
            name='oblique cuts (any angle)',
            line=dict(color=color,width=2),showlegend=True))

    if include_scatter:
        for t in _scatter(X2, y, ds['task']): out.append(t)
    return out, xr, yr


# ══════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def fig_single(ds_name, method, psi, tree_idx, proj_mode='pca'):
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X = ds['X'].astype(np.float32)
    X2, proj = _project2(X, proj_mode)
    part = _load_model(ds_name, method, ds['task'])
    if part is None:
        part = get_partition(method,n_estimators=N_EST,max_samples=psi,random_state=SEED)
        part.fit(X)
    traces,xr,yr = geo_traces_2d(X2, ds, method, part, tree_idx)
    is_inne = method=='inne'
    fig = go.Figure(traces)
    fig.update_layout(**BL, height=520,
        title=f'<b>{PARTITION_NAMES[method]}</b>  ·  {ds_name}  ·  '
              f'{"AD" if ds["task"]=="AD" else "Clustering"}  ·  n={ds["n"]}  ·  '
              f'ψ={psi}  ·  tree #{tree_idx}  ·  {proj}'
              + ('  ·  <i>circles log-scaled</i>' if is_inne else ''),
        xaxis=dict(gridcolor=GRID,zeroline=False,range=xr,title='Component 1',
                   scaleanchor='y',scaleratio=1),
        yaxis=dict(gridcolor=GRID,zeroline=False,range=yr,title='Component 2'))
    return fig


def fig_all4(ds_name, psi, proj_mode='pca'):
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X = ds['X'].astype(np.float32)
    X2, proj = _project2(X, proj_mode)
    methods = ['anne','inne','iforest','sciforest']
    fig = make_subplots(2,2,
        subplot_titles=[PARTITION_NAMES[m] for m in methods],
        horizontal_spacing=0.05, vertical_spacing=0.10)
    ax_map = [(1,1,'','')  ,(1,2,'2','2'),(2,1,'3','3'),(2,2,'4','4')]
    for i,method in enumerate(methods):
        row,col,xs,ys = ax_map[i][0],ax_map[i][1],ax_map[i][2],ax_map[i][3]
        part = _load_model(ds_name, method, ds['task'])
        if part is None:
            part = get_partition(method,n_estimators=N_EST,max_samples=psi,random_state=SEED)
            part.fit(X)
        traces,xr,yr = geo_traces_2d(X2, ds, method, part, 0)
        for t in traces:
            t.showlegend = False
            fig.add_trace(t, row=row, col=col)
        xk = f'xaxis{xs}' if xs else 'xaxis'
        yk = f'yaxis{ys}' if ys else 'yaxis'
        xref = f'x{xs}' if xs else 'x'
        fig.layout[xk].update(range=xr,gridcolor=GRID,zeroline=False,showticklabels=False)
        fig.layout[yk].update(range=yr,gridcolor=GRID,zeroline=False,showticklabels=False,
                               scaleanchor=xref,scaleratio=1)
    fig.update_layout(**BL, height=760,
        title=f'<b>All 4 Partitions</b>  ·  {ds_name}  ·  '
              f'{"AD" if ds["task"]=="AD" else "Clustering"}  ·  '
              f'n={ds["n"]}  ·  ψ={psi}  ·  {proj}')
    return fig


def fig_kernels(ds_name, method, psi):
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X = ds['X'].astype(np.float32); y = ds['y']
    if len(X)>400:
        idx=np.random.RandomState(SEED).choice(len(X),400,replace=False)
        X,y=X[idx],y[idx]
    part=_load_model(ds_name, method, ds['task'])
    if part is None:
        part=get_partition(method,n_estimators=N_EST,max_samples=psi,random_state=SEED)
        part.fit(X)
    order=np.argsort(y)
    K_ik  = part.similarity_ik(X)[np.ix_(order,order)]
    K_idk = part.similarity_idk(X)[np.ix_(order,order)]
    bounds = (np.where(np.diff(y[order])!=0)[0]+0.5).tolist()

    fig=make_subplots(1,2,
        subplot_titles=['IK  —  point-to-point similarity',
                        'IDK  —  normalised distributional similarity'],
        horizontal_spacing=0.12)
    for K,col,cmap in [(K_ik,1,'Viridis'),(K_idk,2,'Cividis')]:
        fig.add_trace(go.Heatmap(z=K,colorscale=cmap,zmin=0,zmax=1,
            showscale=True,
            colorbar=dict(len=0.85,thickness=12,
                          x=0.44 if col==1 else 1.01,
                          title='sim')),row=1,col=col)
        for b in bounds:
            fig.add_shape(dict(type='line',x0=b,x1=b,y0=-0.5,y1=len(K)-0.5,
                               line=dict(color='white',width=1.0)),row=1,col=col)
            fig.add_shape(dict(type='line',y0=b,y1=b,x0=-0.5,x1=len(K)-0.5,
                               line=dict(color='white',width=1.0)),row=1,col=col)
    fig.update_layout(**BL,height=480,
        title=f'<b>Kernel Matrices</b>  ·  {ds_name}  ·  {PARTITION_NAMES[method]}  ·  '
              f'n={len(X)} sorted by class')
    fig.update_xaxes(showticklabels=False,gridcolor=GRID,title='samples →')
    fig.update_yaxes(showticklabels=False,gridcolor=GRID,title='samples →')
    return fig


def fig_scores(ds_name, psi, which, proj_mode='pca'):
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X=ds['X'].astype(np.float32); y=ds['y']
    X2,proj = _project2(X, proj_mode)
    methods=['anne','inne','inne-overlapping','iforest','sciforest']
    rows=max(1,len(which or []))
    kerns = which or ['idk']
    cs_map = {'ik':'RdYlBu_r','idk':'RdYlBu_r'}
    row_labels = {'ik':'IK score  (row 1)','idk':'IDK score  (row 2)'}

    fig=make_subplots(rows,4,
        subplot_titles=[f'{PARTITION_NAMES[m][:14]}' for _ in range(rows) for m in methods],
        horizontal_spacing=0.03,vertical_spacing=0.18)

    idx=np.random.RandomState(SEED).choice(len(X2),min(600,len(X2)),replace=False)

    for mi,method in enumerate(methods):
        col=mi+1
        part=_load_model(ds_name, method, ds['task'])
        if part is None:
            part=get_partition(method,n_estimators=N_EST,max_samples=psi,random_state=SEED)
            part.fit(X)
        for ri,kernel in enumerate(kerns):
            row=ri+1
            sc = (1.0 - part.similarity_ik(X).mean(axis=1)) if kernel=='ik' else part.idk_scores(X)
            cs = 'Blues_r' if kernel=='ik' else 'RdYlBu_r'
            fig.add_trace(go.Scatter(
                x=X2[idx,0].tolist(),y=X2[idx,1].tolist(),mode='markers',
                marker=dict(color=sc[idx].tolist(),colorscale=cs,cmin=0,cmax=1,
                    size=5,opacity=0.85,showscale=(mi==3),
                    colorbar=dict(len=0.35,thickness=10,
                        y=0.83 if row==1 else 0.17,
                        title=dict(text=f'{"IK" if kernel=="ik" else "IDK"}',
                                   font=dict(color=TEXT,size=10))) if mi==3 else None),
                text=[f'{"IK" if kernel=="ik" else "IDK"}={sc[j]:.3f}  '
                      f'class={int(y[j])}' for j in idx],
                hovertemplate='%{text}<extra></extra>',showlegend=False),row=row,col=col)
            if ds['task']=='AD':
                ai=np.where(y==1)[0]
                if len(ai):
                    fig.add_trace(go.Scatter(
                        x=X2[ai,0].tolist(),y=X2[ai,1].tolist(),mode='markers',
                        marker=dict(color='rgba(0,0,0,0)',size=11,symbol='circle-open',
                                    line=dict(color='#ffff00',width=2)),
                        name='true anomaly' if (mi==0 and row==1) else '',
                        showlegend=(mi==0 and row==1)),row=row,col=col)

    for ri,kernel in enumerate(kerns):
        fig.add_annotation(
            x=-0.01, y=1-(ri/rows+1/(2*rows)),
            xref='paper', yref='paper',
            text=f'<b>{"IK" if kernel=="ik" else "IDK"}</b>',
            showarrow=False, font=dict(color='#5af0f7' if kernel=='ik' else '#f7a55a',
                                       size=13), textangle=-90)

    fig.update_xaxes(showticklabels=False,gridcolor=GRID,zeroline=False)
    fig.update_yaxes(showticklabels=False,gridcolor=GRID,zeroline=False)
    fig.update_layout(**BL,height=360*rows+80,
        title=f'<b>Anomaly Scores</b>  ·  {ds_name}  ·  '
              f'{"AD" if ds["task"]=="AD" else "Clustering"}  ·  {proj}  ·  '
              f'dark blue=normal  red=anomalous  ○=true anomaly')
    return fig


def fig_browser(ds_name, proj_mode='pca'):
    ds=DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL),[]
    X=ds['X'].astype(np.float32); y=ds['y']
    X2,proj=_project2(X, proj_mode)
    fig=make_subplots(1,2,
        subplot_titles=['Data scatter (true labels)','Class / anomaly distribution'],
        column_widths=[0.65,0.35])
    for t in _scatter(X2,y,ds['task']): fig.add_trace(t,row=1,col=1)
    classes,counts=np.unique(y,return_counts=True)
    task=ds['task']
    clabels=['normal' if c==0 else 'anomaly' for c in classes] if task=='AD' else [f'class {c}' for c in classes]
    ccolors=['#5af0f7' if c==0 else '#ff5a5a' for c in classes] if task=='AD' else [CPAL[i%len(CPAL)] for i in range(len(classes))]
    fig.add_trace(go.Bar(x=clabels,y=counts.tolist(),marker_color=ccolors,
        text=[f'{c} ({c/len(y)*100:.0f}%)' for c in counts],
        textposition='outside',showlegend=False,textfont=dict(color=TEXT)),row=1,col=2)
    ar=ds.get('anom_rate')
    fig.update_layout(**BL,height=440,
        title=f'<b>{ds_name}</b>  ·  {"AD" if task=="AD" else "Clustering"}  ·  '
              f'C{ds["condition"]} {COND_NAME.get(ds["condition"],"")}  ·  '
              f'n={ds["n"]}  feat={ds["features"]}'+(f'  ·  anom {ar:.1f}%' if ar else '')+
              f'<br><sup>{proj}  ·  shape={ds["shape"]}  density={ds["density"]}  source={ds["source"]}</sup>')
    fig.update_xaxes(gridcolor=GRID,zeroline=False,row=1,col=1)
    fig.update_yaxes(gridcolor=GRID,zeroline=False,row=1,col=1)
    fig.update_xaxes(gridcolor=GRID,row=1,col=2)
    fig.update_yaxes(gridcolor=GRID,title='count',row=1,col=2)
    stats=[
        html.Tr([html.Td('Task',style={'color':MUTED,'width':'100px','paddingRight':'12px'}),
                 html.Td('Anomaly Detection' if task=='AD' else 'Clustering',
                         style={'color':'#f7a55a' if task=='AD' else '#5af7a0','fontWeight':'bold'})]),
        html.Tr([html.Td('Condition'),html.Td(f'{ds["condition"]} — {COND_NAME.get(ds["condition"],"")}',
                         style={'color':COND_COL.get(ds["condition"],'#888')})]),
        html.Tr([html.Td('n (samples)'),html.Td(f'{ds["n"]:,}')]),
        html.Tr([html.Td('Features'),html.Td(str(ds['features']))]),
        html.Tr([html.Td('Classes'),html.Td(str(len(np.unique(y))))]),
        html.Tr([html.Td('Shape'),html.Td(ds['shape'])]),
        html.Tr([html.Td('Density'),html.Td(ds['density'])]),
        html.Tr([html.Td('Dim level'),html.Td(ds['dim_level'])]),
        html.Tr([html.Td('Source'),html.Td(ds['source'])]),
    ]
    if ar: stats.append(html.Tr([html.Td('Anomaly rate'),html.Td(f'{ar:.1f}%',style={'color':'#ff5a5a'})]))
    return fig,stats


def fig_task_lens(ds_name, method, psi, tree_idx, normal_class, proj_mode='pca'):
    """
    Side-by-side comparison:
      left  = clustering view (all classes)
      right = AD view (normal_class = normal, everything else = anomaly)
    """
    ds = DATASETS.get(ds_name)
    if ds is None: return go.Figure().update_layout(**BL)
    X = ds['X'].astype(np.float32)
    y = ds['y']
    X2, proj = _project2(X, proj_mode)

    fig = make_subplots(1, 2,
        subplot_titles=['Clustering view  (all classes)',
                        f'AD view  (class {normal_class} = normal, rest = anomaly)'],
        horizontal_spacing=0.08)

    part = _load_model(ds_name, method, ds['task'])
    if part is None:
        part = get_partition(method, n_estimators=N_EST, max_samples=psi, random_state=SEED)
        part.fit(X)

    # ── Left: clustering view ──
    for t in _scatter(X2, y, 'C', showlegend=False):
        fig.add_trace(t, row=1, col=1)
    gt, xr, yr = geo_traces_2d(X2, {**ds, 'task': 'C'}, method, part, tree_idx, include_scatter=False)
    for t in gt:
        if t.showlegend:
            t.showlegend = False
        fig.add_trace(t, row=1, col=1)

    # ── Right: AD view ──
    y_ad = (y != normal_class).astype(int)
    for t in _scatter(X2, y_ad, 'AD', showlegend=False):
        fig.add_trace(t, row=1, col=2)
    gt_ad, xr_ad, yr_ad = geo_traces_2d(X2, {**ds, 'task': 'AD', 'y': y_ad},
                                         method, part, tree_idx, include_scatter=False)
    for t in gt_ad:
        if t.showlegend:
            t.showlegend = False
        fig.add_trace(t, row=1, col=2)

    fig.update_layout(**BL, height=540,
        title=f'<b>Task Lens</b>  ·  {ds_name}  ·  {PARTITION_NAMES[method]}  ·  '
              f'ψ={psi}  ·  tree #{tree_idx}  ·  {proj}',
        xaxis=dict(gridcolor=GRID, zeroline=False, range=xr, title='X',
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(gridcolor=GRID, zeroline=False, range=yr, title='Y'),
        xaxis2=dict(gridcolor=GRID, zeroline=False, range=xr_ad, title='X',
                    scaleanchor='y2', scaleratio=1),
        yaxis2=dict(gridcolor=GRID, zeroline=False, range=yr_ad, title='Y'))
    return fig


def fig_tradeoff(task_filter='all', cond_filter=0, metric_x='total_time_s'):
    import pandas as pd
    auc_path=os.path.join(ROOT,'results','anomaly_detection','auc_results.csv')
    ari_path=os.path.join(ROOT,'results','clustering','ari_results.csv')
    placeholder=not(os.path.exists(auc_path) or os.path.exists(ari_path))
    if placeholder:
        fig=go.Figure()
        fig.add_annotation(text='Run experiments first:<br><br>'
            '<b>python experiments/run_anomaly.py --fast</b><br>'
            '<b>python experiments/run_clustering.py --fast</b><br><br>'
            'Then reload this tab.',
            xref='paper',yref='paper',x=0.5,y=0.5,showarrow=False,
            font=dict(size=13,color=MUTED),align='center')
        fig.update_layout(**BL,height=400,title='⚖️ Accuracy vs Runtime Trade-off')
        return [fig]

    dfs=[]
    for path,metric in [(auc_path,'auc_mean'),(ari_path,'ari_mean')]:
        if os.path.exists(path):
            df=pd.read_csv(path)
            df['metric_name']=metric
            df['metric_val']=df[metric]
            dfs.append(df)
    if not dfs: return [go.Figure().update_layout(**BL)]
    full=pd.concat(dfs,ignore_index=True)

    if task_filter=='C':  full=full[full['metric_name']=='ari_mean']
    elif task_filter=='AD': full=full[full['metric_name']=='auc_mean']
    if cond_filter>0: full=full[full['condition']==cond_filter]

    fig1=go.Figure()
    for m in ['anne','inne','inne-overlapping','iforest','sciforest']:
        sub=full[full['partition']==m]
        if len(sub)==0: continue
        fig1.add_trace(go.Scatter(
            x=sub[metric_x],y=sub['metric_val'],mode='markers',
            name=PARTITION_NAMES[m],
            marker=dict(color=PC[m],size=9,opacity=0.8,line=dict(color='white',width=0.5)),
            text=[f'<b>{r["dataset"]}</b><br>score={r["metric_val"]:.3f}<br>'
                  f't={r[metric_x]:.3f}s<br>cond={r["condition"]} {COND_NAME.get(r["condition"],"")}'
                  for _,r in sub.iterrows()],
            hovertemplate='%{text}<extra></extra>'))
    fig1.update_layout(**BL,height=420,
        title='<b>Accuracy vs Runtime</b>  ·  hover for details  ·  top-left = best',
        xaxis=dict(gridcolor=GRID,title=metric_x.replace('_',' '),type='log'),
        yaxis=dict(gridcolor=GRID,title='score (AUC or ARI)',range=[0,1.05]))

    fig2=go.Figure()
    for m in ['anne','inne','inne-overlapping','iforest','sciforest']:
        sub=full[full['partition']==m]
        if len(sub)==0: continue
        fig2.add_trace(go.Box(
            x=[f'C{c} {COND_NAME.get(c,"")}' for c in sub['condition']],
            y=sub['metric_val'],name=PARTITION_NAMES[m],
            marker_color=PC[m],line_color=PC[m],fillcolor=PC_RGBA.get(m,'rgba(128,128,128,0.2)'),
            boxpoints='all',jitter=0.3,pointpos=-1.8,
            marker=dict(size=5,opacity=0.7)))
    fig2.update_layout(**BL,height=420,
        title='<b>Score by Condition</b>  ·  each dot = one dataset',
        xaxis=dict(gridcolor=GRID,title='condition',tickangle=-20),
        yaxis=dict(gridcolor=GRID,title='score (AUC or ARI)',range=[0,1.05]),
        boxmode='group')

    pivot=full.groupby(['partition','condition'])['metric_val'].mean().unstack(fill_value=0)
    fig3=go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f'C{c} {COND_NAME.get(c,"")}' for c in pivot.columns],
        y=[PARTITION_NAMES[m] for m in pivot.index],
        colorscale='RdYlGn',zmin=0,zmax=1,
        text=[[f'{v:.2f}' for v in row] for row in pivot.values],
        texttemplate='%{text}',textfont=dict(size=12),
        colorbar=dict(title='mean score')))
    fig3.update_layout(**BL,height=260,
        title='<b>Mean Score Heatmap</b>  ·  partition × condition  ·  green=good  red=bad',
        xaxis=dict(gridcolor=GRID,tickangle=-20),yaxis=dict(gridcolor=GRID))

    return [fig1,fig2,fig3]


# ══════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════
app=dash.Dash(__name__,suppress_callback_exceptions=True)

TS=dict(backgroundColor='#12122a',color=MUTED,border=f'1px solid {BORDER}',
        borderRadius='8px 8px 0 0',padding='10px 18px',fontFamily='monospace',fontSize='12px')
TA={**TS,'backgroundColor':CARD,'color':ACCENT,'borderBottom':f'2px solid {ACCENT}'}

PSI=[{'label':f'ψ = {v}','value':v} for v in [4,8,16,32,64]]
TREES=[{'label':f'tree {i}','value':i} for i in range(8)]
METHS=[{'label':PARTITION_NAMES[m],'value':m} for m in ['anne','inne','inne-overlapping','iforest','sciforest']]

def _dd(id_,opts,val,w='100%'):
    return dcc.Dropdown(id=id_,options=opts,value=val,clearable=False,
        style={'width':w,'backgroundColor':'#12122a','color':'#111',
               'border':f'1px solid {BORDER}','fontFamily':'monospace','fontSize':'11px'})

def _lbl(t): return html.Div(t,style={'color':MUTED,'fontSize':'10px',
    'letterSpacing':'0.5px','marginBottom':'3px','marginTop':'10px'})

def _card(*c,mb=10):
    return html.Div(list(c),style={'backgroundColor':CARD2,'borderRadius':'8px',
        'padding':'12px','border':f'1px solid {BORDER}','marginBottom':f'{mb}px'})

def _head(t,col=ACCENT):
    return html.P(t,style={'color':col,'fontSize':'10px','fontWeight':'bold',
        'letterSpacing':'1.2px','textTransform':'uppercase','margin':'0 0 6px 0'})

def _ds_opts(pool=None, task='all', cond=0, src='all'):
    if pool is None: pool=DATASETS
    def ok(v):
        if task!='all' and v['task']!=task: return False
        if cond!=0 and v['condition']!=cond: return False
        if src=='real' and v['source']=='sklearn_gen': return False
        if src=='synth' and v['source']!='sklearn_gen': return False
        return True
    return [{'label':f'[{"REAL" if v["source"]!="sklearn_gen" else "SYN"}] '
                     f'{k}  —  {v["task"]}  C{v["condition"]} {COND_NAME.get(v["condition"],"")}  '
                     f'n={v["n"]}  feat={v["features"]}','value':k}
            for k,v in pool.items() if ok(v)]


app.layout=html.Div(
    style={'backgroundColor':BG,'minHeight':'100vh','fontFamily':'monospace','padding':'20px 24px'},
    children=[
        html.Div([
            html.H1('IK Partitioning — Visual Explorer',
                    style={'color':'white','margin':'0','fontSize':'18px','letterSpacing':'3px'}),
            html.P('Partition Geometry  ·  All 4 Together  ·  Kernel Matrices  ·  '
                   'Anomaly Scores  ·  Task Lens  ·  Dataset Browser  ·  Trade-off Analysis',
                   style={'color':MUTED,'margin':'4px 0 0 0','fontSize':'10px'}),
        ],style={'marginBottom':'16px','borderBottom':f'1px solid {BORDER}','paddingBottom':'14px'}),

        dcc.Tabs(id='tabs',value='geo',children=[
            dcc.Tab(label='Geometry Lab',    value='geo',      style=TS,selected_style=TA),
            dcc.Tab(label='All 4 Together',  value='all4',     style=TS,selected_style=TA),
            dcc.Tab(label='Kernel View',     value='kernels',  style=TS,selected_style=TA),
            dcc.Tab(label='Anomaly Scores',  value='scores',   style=TS,selected_style=TA),
            dcc.Tab(label='Task Lens',       value='tasklens', style=TS,selected_style=TA),
            dcc.Tab(label='Dataset Browser', value='browser',  style=TS,selected_style=TA),
            dcc.Tab(label='Trade-off',       value='tradeoff', style=TS,selected_style=TA),
            dcc.Tab(label='Winners',         value='winners',  style=TS,selected_style=TA),
        ]),
        html.Div(id='tab-content',
                 style={'backgroundColor':CARD,'borderRadius':'0 10px 10px 10px',
                        'border':f'1px solid {BORDER}','padding':'20px','minHeight':'600px'}),
    ])


@callback(Output('tab-content','children'),Input('tabs','value'))
def render(tab):

    # ── Geometry Lab ──────────────────────────────────────────────────────
    if tab=='geo':
        opts=_ds_opts()
        return html.Div([
            html.Div([
                html.Div([
                    _head('Dataset  (real + synthetic)'),
                    _dd('g-ds',opts,list(DATASETS.keys())[0]),

                    _lbl('Partition method'),
                    _dd('g-m',METHS,'inne'),

                    _lbl('ψ  (subsample size per tree)'),
                    _dd('g-ps',PSI,16),

                    _lbl('Tree index  (one estimator of the ensemble)'),
                    _dd('g-tr',TREES,0),

                    _lbl('Projection'),
                    _dd('g-proj',[],'pca'),

                    html.Div(style={'height':'12px'}),
                    _card(
                        _head('What ψ controls','#f7a55a'),
                        html.P('Small ψ → fewer, larger cells → coarser partition.\n'
                               'Large ψ → more, smaller cells → finer partition.',
                               style={'color':TEXT,'fontSize':'10px','lineHeight':'1.7','margin':'0',
                                      'whiteSpace':'pre-line'}),mb=8),
                    _card(
                        _head('This partition','#5af7a0'),
                        html.Div(id='g-desc',style={'color':TEXT,'fontSize':'10px','lineHeight':'1.7'}),mb=8),
                    _card(
                        _head('When it wins','#f75aab'),
                        html.Div(id='g-when',style={'color':TEXT,'fontSize':'10px','lineHeight':'1.7'}),mb=0),
                ],style={'width':'260px','flexShrink':'0','minWidth':'260px'}),

                html.Div([
                    dcc.Graph(id='g-fig',
                              style={'height':'580px'},
                              config={'displayModeBar':True,'scrollZoom':True}),
                ],style={'flex':'1','minWidth':'0','overflow':'hidden'}),
            ],style={'display':'flex','gap':'16px','alignItems':'flex-start'}),
        ])

    # ── All 4 Together ────────────────────────────────────────────────────
    elif tab=='all4':
        opts=_ds_opts()
        return html.Div([
            html.Div([
                _lbl('Dataset  (real + synthetic, any dimensionality — projected to 2D)'),
                _dd('a-ds',opts,list(DATASETS.keys())[2]),
            ],style={'maxWidth':'700px','marginBottom':'4px'}),
            html.Div([
                html.Div([
                    _lbl('ψ  (subsample size)'),
                    _dd('a-ps',PSI,16,'130px'),
                ],style={'marginRight':'16px'}),
                html.Div([
                    _lbl('Projection'),
                    _dd('a-proj',[],'pca','200px'),
                ]),
            ],style={'display':'flex','marginBottom':'12px'}),
            dcc.Graph(id='a-fig',style={'height':'780px'},
                      config={'displayModeBar':True,'scrollZoom':True}),
            _card(html.Div([
                html.Span('Voronoi: ',style={'fontWeight':'bold','color':PC['anne']}),
                html.Span('nearest-centroid lines.  ',style={'color':TEXT}),
                html.Span('Hypersphere: ',style={'fontWeight':'bold','color':PC['inne']}),
                html.Span('circles (log-scaled), big = sparse = anomaly.  ',style={'color':TEXT}),
                html.Span('iForest: ',style={'fontWeight':'bold','color':PC['iforest']}),
                html.Span('H/V lines only.  ',style={'color':TEXT}),
                html.Span('SCiForest: ',style={'fontWeight':'bold','color':PC['sciforest']}),
                html.Span('oblique at any angle.',style={'color':TEXT}),
            ],style={'fontSize':'11px'}),mb=0),
        ])

    # ── Kernel View ───────────────────────────────────────────────────────
    elif tab=='kernels':
        opts=_ds_opts()
        return html.Div([
            html.Div([
                _lbl('Dataset'),
                _dd('k-ds',opts,list(DATASETS.keys())[0]),
            ],style={'maxWidth':'700px','marginBottom':'4px'}),
            html.Div([
                _lbl('Partition method'),
                _dd('k-m',METHS,'anne','300px'),
            ],style={'marginBottom':'4px'}),
            html.Div([
                _lbl('ψ'),
                _dd('k-ps',PSI,16,'130px'),
            ],style={'marginBottom':'12px'}),
            _card(
                _head('What am I looking at?'),
                html.Div([
                    html.P('Each cell (i,j) shows how similar samples i and j are under this partition. '
                           'Samples are sorted by their true class label. White lines = class boundaries.',
                           style={'color':TEXT,'fontSize':'11px','lineHeight':'1.7','margin':'0 0 8px 0'}),
                    html.P([
                        html.Span('IK (left, purple): ',style={'color':'#a07af7','fontWeight':'bold'}),
                        html.Span('κ(x,y) = probability x and y land in the same cell.  '
                                  'Raw co-occurrence.  ',style={'color':TEXT}),
                    ],style={'fontSize':'11px','margin':'0 0 4px 0'}),
                    html.P([
                        html.Span('IDK (right, yellow): ',style={'color':'#f7e05a','fontWeight':'bold'}),
                        html.Span('same idea but normalised so each point has self-similarity = 1.  '
                                  'Gives more contrast between classes.',style={'color':TEXT}),
                    ],style={'fontSize':'11px','margin':'0 0 4px 0'}),
                    html.P([
                        html.Span('Good partition → ',style={'color':'#5af7a0','fontWeight':'bold'}),
                        html.Span('bright diagonal blocks (same class = high similarity).  '
                                  'Off-diagonal should be dark.  '
                                  'Compare IK vs IDK: IDK usually has cleaner blocks.',
                                  style={'color':TEXT}),
                    ],style={'fontSize':'11px','margin':'0'}),
                ]),mb=10),
            dcc.Graph(id='k-fig',style={'height':'490px'},config={'displayModeBar':True}),
        ])

    # ── Anomaly Scores ────────────────────────────────────────────────────
    elif tab=='scores':
        opts=_ds_opts()
        return html.Div([
            html.Div([
                _lbl('Dataset'),
                _dd('s-ds',opts,list(DATASETS.keys())[0]),
            ],style={'maxWidth':'700px','marginBottom':'4px'}),
            html.Div([
                html.Div([
                    _lbl('ψ'),
                    _dd('s-ps',PSI,16,'130px'),
                ],style={'marginRight':'16px'}),
                html.Div([
                    _lbl('Projection'),
                    _dd('s-proj',[],'pca','200px'),
                ]),
            ],style={'display':'flex','marginBottom':'4px'}),
            html.Div([
                _lbl('Which scores to show'),
                dcc.Checklist(id='s-show',
                    options=[{'label':'  IK score  (1 − mean point-to-point similarity)',  'value':'ik'},
                             {'label':'  IDK score  (1 − similarity to global distribution)','value':'idk'}],
                    value=['idk'], labelStyle={'display':'block','marginBottom':'4px'},
                    style={'color':TEXT,'fontSize':'11px','marginTop':'4px'}),
            ],style={'marginBottom':'12px'}),
            _card(html.Div([
                html.Span('IK row (blue scale): ',style={'color':'#5af0f7','fontWeight':'bold'}),
                html.Span('dark blue = normal, light = anomalous.  ',style={'color':TEXT}),
                html.Span('IDK row (red-blue scale): ',style={'color':'#f7a55a','fontWeight':'bold'}),
                html.Span('blue = normal, red = anomalous.  ',style={'color':TEXT}),
                html.Span('○ = true anomaly.  ',style={'color':'#ffff00','fontWeight':'bold'}),
                html.Span('Hover any point to see exact score.',style={'color':MUTED}),
            ],style={'fontSize':'11px'}),mb=10),
            dcc.Graph(id='s-fig',config={'displayModeBar':True,'scrollZoom':True}),
        ])

    # ── Task Lens ─────────────────────────────────────────────────────────
    elif tab=='tasklens':
        opts = _ds_opts()
        return html.Div([
            html.Div([
                html.Div([
                    _lbl('Dataset'),
                    _dd('tl-ds', opts, list(DATASETS.keys())[0]),
                ], style={'maxWidth':'500px', 'marginRight':'16px'}),
                html.Div([
                    _lbl('Partition method'),
                    _dd('tl-m', METHS, 'inne', '220px'),
                ], style={'marginRight':'16px'}),
                html.Div([
                    _lbl('ψ'),
                    _dd('tl-ps', PSI, 16, '100px'),
                ], style={'marginRight':'16px'}),
                html.Div([
                    _lbl('Tree index'),
                    _dd('tl-tr', TREES, 0, '100px'),
                ], style={'marginRight':'16px'}),
                html.Div([
                    _lbl('Normal class (for AD view)'),
                    _dd('tl-norm', [], 0, '120px'),
                ]),
            ], style={'display':'flex', 'flexWrap':'wrap', 'marginBottom':'4px'}),
            html.Div([
                html.Div([
                    _lbl('Projection'),
                    _dd('tl-proj', [], 'pca', '220px'),
                ]),
            ], style={'marginBottom':'12px'}),
            _card(
                _head('What is the Task Lens?'),
                html.Div([
                    html.P('This view compares how the same dataset looks under two different task framings:',
                           style={'color':TEXT,'fontSize':'11px','lineHeight':'1.7','margin':'0 0 8px 0'}),
                    html.P([
                        html.Span('Left (Clustering): ',style={'color':'#5af7a0','fontWeight':'bold'}),
                        html.Span('every true class gets its own colour.  '
                                  'Useful for seeing how well the partition geometry separates natural clusters.',
                                  style={'color':TEXT}),
                    ],style={'fontSize':'11px','margin':'0 0 4px 0'}),
                    html.P([
                        html.Span('Right (Anomaly Detection): ',style={'color':'#ff5a5a','fontWeight':'bold'}),
                        html.Span('one selected class is treated as "normal" (cyan circles) and '
                                  'everything else as "anomaly" (red crosses).  '
                                  'Useful for evaluating how the partition would behave if the problem were framed as AD.',
                                  style={'color':TEXT}),
                    ],style={'fontSize':'11px','margin':'0'}),
                ]),mb=10),
            dcc.Graph(id='tl-fig', style={'height':'560px'},
                      config={'displayModeBar':True,'scrollZoom':True}),
        ])

    # ── Dataset Browser ───────────────────────────────────────────────────
    elif tab=='browser':
        return html.Div([
            html.Div([
                html.Div([
                    _head('Filter datasets'),
                    _lbl('Task'),
                    _dd('b-task',[{'label':'All tasks','value':'all'},
                                  {'label':'Clustering only','value':'C'},
                                  {'label':'Anomaly detection only','value':'AD'}],'all'),
                    _lbl('Condition'),
                    _dd('b-cond',[{'label':'All conditions','value':0}]+
                        [{'label':f'C{c} — {COND_NAME[c]}','value':c} for c in range(1,8)],0),
                    _lbl('Source'),
                    _dd('b-src',[{'label':'All (real + synthetic)','value':'all'},
                                 {'label':'Real data only','value':'real'},
                                 {'label':'Synthetic only','value':'synth'}],'all'),
                    html.Div(id='b-count',style={'color':MUTED,'fontSize':'10px',
                                                  'margin':'8px 0 4px 0'}),
                    _lbl('Select dataset'),
                    _dd('b-ds',_ds_opts(),'iris'),
                    html.Div(style={'height':'16px'}),
                    _head('Dataset stats'),
                    html.Table(id='b-stats',
                               style={'fontSize':'11px','lineHeight':'2.1',
                                      'color':TEXT,'width':'100%'}),
                ],style={'width':'260px','flexShrink':'0','minWidth':'260px'}),

                html.Div([
                    html.Div([
                        _lbl('Projection'),
                        _dd('b-proj',[],'pca','220px'),
                    ],style={'marginBottom':'8px'}),
                    dcc.Graph(id='b-fig',
                              style={'height':'480px'},
                              config={'displayModeBar':True,'scrollZoom':True,
                                      'staticPlot':False}),
                ],style={'flex':'1','minWidth':'0','overflow':'hidden'}),
            ],style={'display':'flex','gap':'16px','alignItems':'flex-start'}),
        ])

    # ── Trade-off ─────────────────────────────────────────────────────────
    elif tab=='tradeoff':
        return html.Div([
            html.Div([
                html.Div([
                    _lbl('Task filter'),
                    _dd('t-task',[{'label':'All tasks','value':'all'},
                                  {'label':'Clustering (ARI)','value':'C'},
                                  {'label':'Anomaly detection (AUC)','value':'AD'}],'all','200px'),
                ],style={'marginRight':'16px'}),
                html.Div([
                    _lbl('Condition filter'),
                    _dd('t-cond',[{'label':'All conditions','value':0}]+
                        [{'label':f'C{c} — {COND_NAME[c]}','value':c} for c in range(1,8)],0,'200px'),
                ],style={'marginRight':'16px'}),
                html.Div([
                    _lbl('X-axis metric'),
                    _dd('t-xmet',[{'label':'Total runtime (s)','value':'total_time_s'},
                                  {'label':'Fit time (s)','value':'fit_time_s'},
                                  {'label':'Transform time (s)','value':'transform_time_s'}],
                        'total_time_s','200px'),
                ]),
            ],style={'display':'flex','flexWrap':'wrap','marginBottom':'14px','alignItems':'flex-end'}),
            html.Div(id='t-figs'),
        ])

    # ── Winners ───────────────────────────────────────────────────────────
    elif tab=='winners':
        return html.Div([
            html.Div(id='w-content'),
        ])


# ── Callbacks ──────────────────────────────────────────────────────────────

@callback(Output('g-proj','options'),Output('g-proj','value'),
          Input('g-ds','value'))
def cb_geo_proj_opts(ds_name):
    opts, default = _proj_opts(ds_name)
    return opts, default


@callback(Output('g-fig','figure'),Output('g-desc','children'),Output('g-when','children'),
          Input('g-ds','value'),Input('g-m','value'),Input('g-ps','value'),
          Input('g-tr','value'),Input('g-proj','value'))
def cb_geo(ds,m,ps,tr,proj):
    from src.partitions import PARTITION_NAMES as PN
    PDESC={'anne':'Voronoi cells: each of the ψ random centroids owns all data closer to it than others. Cells are large in sparse regions, small in dense ones.',
           'inne':'Hypersphere partition: each centroid gets a ball whose radius = distance to its nearest neighbour. Big ball = sparse = likely anomaly. Key density-adaptive property.',
           'iforest':'Axis-parallel cuts: recursive H/V splits only. Fast and simple, but fails when clusters are diagonal, curved, or nested.',
           'sciforest':'Oblique hyperplane splits: each cut uses a random linear combination of features so lines go diagonal. More flexible than iForest.'}
    PWHEN={'anne':'Best when clusters are compact and well-separated. Good baseline for all 7 conditions.',
           'inne':'Best for varying-density data (C5) and crescent/irregular shapes (C3). Survey paper: iForest FAILS on crescent, iNNE achieves perfect AUC.',
           'iforest':'Best for axis-aligned data and large-scale problems (C7). Avoid on diagonal clusters (C2), curves (C3), nested shapes (C4).',
           'sciforest':'Best for elongated/diagonal clusters (C2) and high-dimensional data (C6). Better than iForest when data has rotational structure.'}
    return fig_single(ds,m,ps or 16,tr or 0,proj or 'pca'), PDESC.get(m,''), PWHEN.get(m,'')


@callback(Output('a-proj','options'),Output('a-proj','value'),
          Input('a-ds','value'))
def cb_all4_proj_opts(ds_name):
    opts, default = _proj_opts(ds_name)
    return opts, default


@callback(Output('a-fig','figure'),Input('a-ds','value'),Input('a-ps','value'),Input('a-proj','value'))
def cb_all4(ds,ps,proj): return fig_all4(ds,ps or 16,proj or 'pca')


@callback(Output('k-fig','figure'),
          Input('k-ds','value'),Input('k-m','value'),Input('k-ps','value'))
def cb_ker(ds,m,ps): return fig_kernels(ds,m,ps or 16)


@callback(Output('s-proj','options'),Output('s-proj','value'),
          Input('s-ds','value'))
def cb_scores_proj_opts(ds_name):
    opts, default = _proj_opts(ds_name)
    return opts, default


@callback(Output('s-fig','figure'),
          Input('s-ds','value'),Input('s-ps','value'),Input('s-show','value'),Input('s-proj','value'))
def cb_scores(ds,ps,show,proj): return fig_scores(ds,ps or 16,show or ['idk'],proj or 'pca')


@callback(Output('tl-proj','options'),Output('tl-proj','value'),
          Output('tl-norm','options'),Output('tl-norm','value'),
          Input('tl-ds','value'))
def cb_tasklens_opts(ds_name):
    ds = DATASETS.get(ds_name)
    proj_opts, proj_default = _proj_opts(ds_name)
    if ds is None:
        return proj_opts, proj_default, [{'label':'0','value':0}], 0
    classes = sorted(np.unique(ds['y']).tolist())
    norm_opts = [{'label':f'class {c}','value':c} for c in classes]
    norm_default = classes[0] if classes else 0
    return proj_opts, proj_default, norm_opts, norm_default


@callback(Output('tl-fig','figure'),
          Input('tl-ds','value'),Input('tl-m','value'),Input('tl-ps','value'),
          Input('tl-tr','value'),Input('tl-norm','value'),Input('tl-proj','value'))
def cb_tasklens(ds,m,ps,tr,norm,proj):
    return fig_task_lens(ds,m,ps or 16,tr or 0,norm if norm is not None else 0,proj or 'pca')


@callback(Output('b-proj','options'),Output('b-proj','value'),
          Input('b-ds','value'))
def cb_browser_proj_opts(ds_name):
    opts, default = _proj_opts(ds_name)
    return opts, default


@callback(Output('b-fig','figure'),Output('b-stats','children'),
          Output('b-ds','options'),Output('b-count','children'),
          Input('b-ds','value'),Input('b-task','value'),
          Input('b-cond','value'),Input('b-src','value'),Input('b-proj','value'))
def cb_browser(ds_name,task,cond,src,proj):
    filtered={k:v for k,v in DATASETS.items()
              if (task=='all' or v['task']==task)
              and (cond==0 or v['condition']==cond)
              and (src=='all' or (src=='real' and v['source']!='sklearn_gen')
                   or (src=='synth' and v['source']=='sklearn_gen'))}
    opts=_ds_opts(filtered)
    if ds_name not in filtered and filtered: ds_name=next(iter(filtered))
    fig,stats=fig_browser(ds_name,proj or 'pca')
    count=f'{len(filtered)} datasets match filter'
    return fig,stats,opts,count


@callback(Output('t-figs','children'),
          Input('t-task','value'),Input('t-cond','value'),Input('t-xmet','value'))
def cb_tradeoff(task,cond,xmet):
    figs=fig_tradeoff(task or 'all',cond or 0,xmet or 'total_time_s')
    return [dcc.Graph(figure=f,config={'displayModeBar':True},
                      style={'marginBottom':'12px'}) for f in figs]


# ══════════════════════════════════════════════════════════════════════════
# WINNERS TAB
# ══════════════════════════════════════════════════════════════════════════

def _load_winners():
    """Load result CSVs and return (ad_winners, cl_winners) dicts."""
    ad_path = os.path.join(ROOT, 'results', 'anomaly_detection', 'auc_results.csv')
    cl_path = os.path.join(ROOT, 'results', 'clustering', 'ari_results.csv')
    ad_win, cl_win = {}, {}
    if os.path.exists(ad_path):
        df = pd.read_csv(ad_path)
        # Use the most recent run per (dataset, partition, kernel)
        df = df.sort_values('timestamp').drop_duplicates(
            ['dataset', 'partition', 'kernel'], keep='last'
        )
        # Best partition per dataset (max auc_mean across both kernels)
        best = df.loc[df.groupby('dataset')['auc_mean'].idxmax()]
        for _, r in best.iterrows():
            ad_win[r['dataset']] = {
                'partition': r['partition'],
                'partition_name': PARTITION_NAMES.get(r['partition'], r['partition']),
                'kernel': r['kernel'],
                'score': r['auc_mean'],
                'score_std': r['auc_std'],
                'time': r['total_time_s'],
                'condition': r['condition'],
            }
    if os.path.exists(cl_path):
        df = pd.read_csv(cl_path)
        df = df.sort_values('timestamp').drop_duplicates(
            ['dataset', 'partition', 'kernel'], keep='last'
        )
        best = df.loc[df.groupby('dataset')['ari_mean'].idxmax()]
        for _, r in best.iterrows():
            cl_win[r['dataset']] = {
                'partition': r['partition'],
                'partition_name': PARTITION_NAMES.get(r['partition'], r['partition']),
                'kernel': r['kernel'],
                'score': r['ari_mean'],
                'score_std': r['ari_std'],
                'time': r['total_time_s'],
                'condition': r['condition'],
            }
    return ad_win, cl_win


@callback(Output('w-content','children'),
          Input('tabs','value'))
def cb_winners(tab):
    if tab != 'winners':
        return []
    ad_win, cl_win = _load_winners()
    if not ad_win and not cl_win:
        return html.Div([
            html.P('No results found. Run experiments first:', style={'color': MUTED}),
            html.P('python experiments/run_anomaly.py', style={'color': ACCENT}),
            html.P('python experiments/run_clustering.py', style={'color': ACCENT}),
        ])

    rows = []
    for name in sorted(set(list(ad_win.keys()) + list(cl_win.keys()))):
        ad = ad_win.get(name)
        cl = cl_win.get(name)
        ds = DATASETS.get(name)
        cond = int(ds['condition']) if ds else (ad['condition'] if ad else cl['condition'])
        task_badge = []
        if ad:
            task_badge.append(html.Span('AD', style={
                'backgroundColor': '#ff5a5a22', 'color': '#ff5a5a',
                'padding': '2px 8px', 'borderRadius': '4px', 'fontSize': '10px',
                'marginRight': '6px'
            }))
        if cl:
            task_badge.append(html.Span('C', style={
                'backgroundColor': '#5af7a022', 'color': '#5af7a0',
                'padding': '2px 8px', 'borderRadius': '4px', 'fontSize': '10px'
            }))
        rows.append(html.Tr([
            html.Td(name, style={'color': TEXT, 'padding': '8px', 'borderBottom': f'1px solid {BORDER}'}),
            html.Td(task_badge, style={'padding': '8px', 'borderBottom': f'1px solid {BORDER}'}),
            html.Td(f'C{cond}', style={'color': MUTED, 'padding': '8px', 'borderBottom': f'1px solid {BORDER}'}),
            html.Td([
                html.Span(ad['partition_name'] if ad else '-', style={'color': PC.get(ad['partition'], TEXT) if ad else MUTED}),
                html.Span(f'  {ad["score"]:.3f}' if ad else '', style={'color': MUTED, 'fontSize': '10px'}),
            ], style={'padding': '8px', 'borderBottom': f'1px solid {BORDER}'}),
            html.Td([
                html.Span(cl['partition_name'] if cl else '-', style={'color': PC.get(cl['partition'], TEXT) if cl else MUTED}),
                html.Span(f'  {cl["score"]:.3f}' if cl else '', style={'color': MUTED, 'fontSize': '10px'}),
            ], style={'padding': '8px', 'borderBottom': f'1px solid {BORDER}'}),
        ]))

    return html.Div([
        html.Div([
            _head('Experiment Winners'),
            html.P('Best partition per dataset according to the latest experiment results. '
                   'Go to Geometry Lab and select the dataset + winner to inspect why it won.',
                   style={'color': TEXT, 'fontSize': '11px', 'margin': '0 0 12px 0'}),
        ], style={'marginBottom': '10px'}),
        html.Table([
            html.Thead(html.Tr([
                html.Th('Dataset', style={'color': ACCENT, 'textAlign': 'left', 'padding': '8px', 'borderBottom': f'2px solid {ACCENT}'}),
                html.Th('Task', style={'color': ACCENT, 'textAlign': 'left', 'padding': '8px', 'borderBottom': f'2px solid {ACCENT}'}),
                html.Th('Cond', style={'color': ACCENT, 'textAlign': 'left', 'padding': '8px', 'borderBottom': f'2px solid {ACCENT}'}),
                html.Th('AD Winner  (AUC)', style={'color': ACCENT, 'textAlign': 'left', 'padding': '8px', 'borderBottom': f'2px solid {ACCENT}'}),
                html.Th('CL Winner  (ARI)', style={'color': ACCENT, 'textAlign': 'left', 'padding': '8px', 'borderBottom': f'2px solid {ACCENT}'}),
            ])),
            html.Tbody(rows),
        ], style={'width': '100%', 'fontSize': '12px', 'borderCollapse': 'collapse'}),
    ])


if __name__=='__main__':
    print('\n  IK Partition Visualizer → http://127.0.0.1:8054\n')
    app.run(debug=False,use_reloader=False,port=8054)
