"""
notebooks/eda.py  —  Interactive EDA Dashboard for IK Partitioning Study
=========================================================================
Run with:
    python notebooks/eda.py

Opens at: http://127.0.0.1:8053

Tabs:
  1. Overview       — dataset counts, condition coverage, task balance
  2. Scatter        — 2D / PCA scatter per dataset coloured by class
  3. Distributions  — feature stats, n vs features, anomaly rates
  4. Conditions     — deep-dive per condition: why it matters, datasets inside
  5. Coverage Check — gap analysis: do we have enough variety?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_blobs, make_moons, make_circles,
    make_gaussian_quantiles, make_classification,
)

# ── Theme ──────────────────────────────────────────────────────────────────
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

COND_COLORS = {
    1: "#4a90d9", 2: "#9b7ff7", 3: "#f79b6a",
    4: "#f76aab", 5: "#6af7a0", 6: "#f7e06a", 7: "#aaaaaa",
}
COND_NAMES = {
    1: "Spherical",        2: "Elongated",
    3: "Crescent",         4: "Nested",
    5: "Varying density",  6: "High-dimensional",
    7: "Large (efficiency)",
}
COND_WHY = {
    1: "Baseline — all 4 partitions should perform similarly here. "
       "If any partition fails on spherical clusters, it has a fundamental problem.",
    2: "iForest (axis-parallel) struggles with elongated shapes — its rectangular "
       "cuts create overextended partitions. Random hyperplane should win here.",
    3: "Directly from the 2025 survey paper (Table 3): iForest FAILS on crescent "
       "shapes and different densities. iNNE wins by adapting ball sizes locally.",
    4: "Hardest case for all hyperplane-based partitions. Only the hypersphere "
       "(iNNE) and Voronoi (aNNE) adapt geometrically to nested structures.",
    5: "Key test for hypersphere advantage: iNNE assigns large balls in sparse "
       "regions and small balls in dense regions. iForest has no such adaptation.",
    6: "Tests the curse of dimensionality. Voronoi degrades in high-dim spaces. "
       "Random hyperplane (SCiForest) was specifically designed for this.",
    7: "Computational efficiency test. iForest is O(n log n) and the fastest. "
       "Voronoi is most expensive. Differences only visible at scale.",
}
SHAPE_COLORS = {
    "spherical":  "#4a90d9",
    "elliptical": "#9b7ff7",
    "crescent":   "#f79b6a",
    "irregular":  "#f76aab",
    "nested":     "#6af7a0",
    "mixed":      "#f7e06a",
}
SRC_COLORS = {
    "sklearn":     "#4a90d9",
    "sklearn_gen": "#6af7a0",
    "UCI":         "#9b7ff7",
    "ADBench":     "#f7e06a",
}

base_layout = dict(
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    font=dict(color=TEXT, size=11, family="monospace"),
    margin=dict(l=50, r=20, t=50, b=50),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, font=dict(color=TEXT)),
)


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def norm(X):
    return MinMaxScaler().fit_transform(np.nan_to_num(X.astype(float)))

def encode(y):
    return LabelEncoder().fit_transform(np.array(y).ravel())

def make_ad(X_in, frac=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n_out = max(1, int(len(X_in) * frac / (1 - frac)))
    lo, hi = X_in.min(0), X_in.max(0)
    span = hi - lo
    X_out = rng.uniform(lo - 0.5*span, hi + 0.5*span, (n_out, X_in.shape[1]))
    X = np.vstack([X_in, X_out])
    y = np.array([0]*len(X_in) + [1]*n_out)
    return X, y

def build_datasets():
    rows = []

    def add(name, X, y, task, shape, density, dim, size, src, cond):
        X2 = norm(X); y2 = encode(y)
        n_cl = int(len(np.unique(y2)))
        anom = round(float(y2.mean())*100, 1) if task == 'AD' else None
        rows.append(dict(
            name=name, X=X2, y=y2, task=task,
            shape=shape, density=density,
            dim_level=dim, size_level=size,
            source=src, condition=cond,
            n=int(X2.shape[0]), features=int(X2.shape[1]),
            n_classes=n_cl, anom_rate=anom,
        ))

    # ── CONDITION 1 — Spherical ──
    d = load_iris()
    add('iris', d.data, d.target,
        'C','spherical','uniform','low','small','sklearn',1)
    d = load_breast_cancer()
    add('breast_cancer', d.data, d.target,
        'AD','spherical','uniform','mid','medium','sklearn',1)
    X,y = make_blobs(500,3,cluster_std=0.8,random_state=42)
    add('syn_blobs_small',X,y,'C','spherical','uniform','low','small','sklearn_gen',1)
    X,y = make_blobs(3000,4,cluster_std=1.0,random_state=42)
    add('syn_blobs_medium',X,y,'C','spherical','uniform','low','medium','sklearn_gen',1)

    # ── CONDITION 2 — Elongated ──
    d = load_wine()
    add('wine', d.data, d.target,
        'C','elliptical','uniform','low','small','sklearn',2)
    X,y = make_blobs(500,3,cluster_std=[3.,.5,2.],random_state=42)
    add('syn_elongated_small',X,y,'C','elliptical','uniform','low','small','sklearn_gen',2)
    X,y = make_blobs(800,3,cluster_std=[4.,.3,2.5],random_state=42)
    add('syn_elongated_medium',X,y,'C','elliptical','uniform','low','medium','sklearn_gen',2)

    # ── CONDITION 3 — Crescent ──
    X,y = make_moons(500,noise=0.05,random_state=42)
    add('syn_moons_small',X,y,'C','crescent','uniform','low','small','sklearn_gen',3)
    X,y = make_moons(1000,noise=0.10,random_state=42)
    add('syn_moons_medium',X,y,'C','crescent','uniform','low','medium','sklearn_gen',3)
    X_in,_ = make_moons(950,noise=0.05,random_state=42)
    X_ad,y_ad = make_ad(X_in, 0.05)
    add('syn_moons_ad',X_ad,y_ad,'AD','crescent','uniform','low','medium','sklearn_gen',3)

    # ── CONDITION 4 — Nested ──
    X,y = make_circles(500,noise=0.05,factor=0.4,random_state=42)
    add('syn_circles_small',X,y,'C','nested','uniform','low','small','sklearn_gen',4)
    X,y = make_circles(800,noise=0.08,factor=0.5,random_state=42)
    add('syn_circles_medium',X,y,'C','nested','uniform','low','medium','sklearn_gen',4)
    X,y = make_gaussian_quantiles(n_samples=500,n_features=2,n_classes=3,random_state=42)
    add('syn_gauss_quantiles',X,y,'C','nested','uniform','low','small','sklearn_gen',4)

    # ── CONDITION 5 — Varying density ──
    X1,_ = make_blobs(400,centers=[[0,0]],cluster_std=0.3,random_state=42)
    X2,_ = make_blobs(100,centers=[[6,6]],cluster_std=2.5,random_state=42)
    X=np.vstack([X1,X2]); y=np.array([0]*400+[1]*100)
    add('syn_density_2d',X,y,'C','mixed','varying','low','small','sklearn_gen',5)
    X1,_=make_blobs(300,centers=[[0,0,0]],cluster_std=0.2,random_state=42)
    X2,_=make_blobs(300,centers=[[5,5,5]],cluster_std=1.5,random_state=42)
    X3,_=make_blobs(400,centers=[[10,0,5]],cluster_std=3.0,random_state=42)
    X=np.vstack([X1,X2,X3]); y=np.array([0]*300+[1]*300+[2]*400)
    add('syn_density_3d',X,y,'C','mixed','varying','low','medium','sklearn_gen',5)

    # ── CONDITION 6 — High-dim ──
    d = load_digits()
    add('digits',d.data,d.target,'C','mixed','uniform','high','large','sklearn',6)
    X,y = make_classification(1000,n_features=50,n_informative=20,n_redundant=10,
                              n_classes=3,n_clusters_per_class=1,random_state=42)
    add('syn_highdim_50',X,y,'C','spherical','uniform','high','medium','sklearn_gen',6)
    X,y = make_classification(1000,n_features=100,n_informative=30,n_redundant=20,
                              n_classes=4,n_clusters_per_class=1,random_state=42)
    add('syn_highdim_100',X,y,'C','spherical','uniform','vhigh','medium','sklearn_gen',6)

    # ── CONDITION 7 — Large ──
    X,y = make_blobs(10000,5,cluster_std=1.5,random_state=42)
    add('syn_large_10k',X,y,'AD','spherical','uniform','low','large','sklearn_gen',7)
    X,y = make_blobs(20000,5,cluster_std=1.5,random_state=42)
    add('syn_large_20k',X,y,'AD','spherical','uniform','low','large','sklearn_gen',7)

    # ── ADBench (if downloaded) ──
    adbench_map = {
        'breastw':'4_breastw.npz','wbc':'42_WBC.npz',
        'ionosphere':'18_Ionosphere.npz','vowels':'40_vowels.npz',
        'lympho':'21_Lymphography.npz','thyroid':'38_thyroid.npz',
        'cardio':'6_cardio.npz','satellite':'30_satellite.npz',
        'musk':'25_musk.npz','optdigits':'26_optdigits.npz',
        'pendigits':'28_pendigits.npz','annthyroid':'2_annthyroid.npz',
        'shuttle':'32_shuttle.npz','waveform':'41_Waveform.npz',
        'letter':'20_letter.npz',
    }
    adbench_meta = [
        ('breastw',   'AD','spherical', 'uniform','low', 'small', 'ADBench',1),
        ('wbc',       'AD','spherical', 'uniform','mid', 'small', 'ADBench',1),
        ('ionosphere','AD','elliptical','uniform','mid', 'small', 'ADBench',2),
        ('vowels',    'AD','irregular', 'uniform','low', 'medium','ADBench',3),
        ('lympho',    'AD','irregular', 'sparse', 'low', 'small', 'ADBench',3),
        ('thyroid',   'AD','mixed',     'varying','low', 'medium','ADBench',5),
        ('cardio',    'AD','mixed',     'varying','mid', 'medium','ADBench',5),
        ('waveform',  'C', 'mixed',     'varying','mid', 'medium','ADBench',5),
        ('satellite', 'AD','mixed',     'varying','high','large', 'ADBench',6),
        ('musk',      'AD','mixed',     'uniform','vhigh','medium','ADBench',6),
        ('optdigits', 'AD','mixed',     'uniform','high','medium','ADBench',6),
        ('pendigits', 'AD','mixed',     'uniform','low', 'large', 'ADBench',6),
        ('annthyroid','AD','mixed',     'varying','low', 'large', 'ADBench',7),
        ('shuttle',   'AD','mixed',     'uniform','low', 'large', 'ADBench',7),
    ]
    save_dir = 'data/anomaly_detection'
    for name, task, shape, density, dim, size, src, cond in adbench_meta:
        fname = adbench_map.get(name)
        if fname:
            path = os.path.join(save_dir, fname)
            if os.path.exists(path):
                try:
                    d = np.load(path, allow_pickle=True)
                    add(name, d['X'].astype(float),
                        d['y'].ravel().astype(int),
                        task, shape, density, dim, size, src, cond)
                except Exception:
                    pass

    # ── UCI (if ucimlrepo) ──
    try:
        from ucimlrepo import fetch_ucirepo
        uci_list = [
            (15,  'wbc_uci',    'AD','spherical', 'uniform','mid','small', 1),
            (42,  'glass',      'C', 'elliptical','uniform','low','small', 2),
            (149, 'vehicle',    'C', 'elliptical','uniform','mid','medium',2),
            (39,  'ecoli',      'C', 'mixed',     'varying','low','small', 5),
            (110, 'yeast',      'C', 'mixed',     'varying','low','medium',5),
            (33,  'dermatology','C', 'mixed',     'uniform','high','small',6),
            (59,  'letter',     'C', 'mixed',     'uniform','low','large', 7),
        ]
        for uid, name, task, shape, density, dim, size, cond in uci_list:
            try:
                ds = fetch_ucirepo(id=uid)
                X = ds.data.features.values.astype(float)
                y = ds.data.targets.values.ravel()
                add(name, X, y, task, shape, density, dim, size, 'UCI', cond)
            except Exception:
                pass
    except ImportError:
        pass

    return rows

# ── UI helpers ─────────────────────────────────────────────────────────────

def card(children, mb=16, extra=None):
    s = {"backgroundColor": CARD, "borderRadius": "12px",
         "padding": "18px", "border": f"1px solid {BORDER}",
         "marginBottom": f"{mb}px"}
    if extra:
        s.update(extra)
    return html.Div(children, style=s)

def sec(text, color=C_PURPLE):
    return html.H3(text, style={
        "color": color, "marginBottom": "12px", "marginTop": "0",
        "fontSize": "12px", "letterSpacing": "1.5px",
        "textTransform": "uppercase",
    })

def badge(text, color, text_color="white"):
    return html.Span(text, style={
        "backgroundColor": color, "color": text_color,
        "padding": "3px 10px", "borderRadius": "10px",
        "fontSize": "10px", "marginRight": "6px", "fontWeight": "bold",
    })

def big_stat(label, value, color):
    return html.Div([
        html.P(label, style={"color": MUTED, "margin": "0", "fontSize": "10px",
                              "letterSpacing": "0.5px"}),
        html.H2(str(value), style={"color": color, "margin": "4px 0 0 0",
                                    "fontSize": "28px", "fontWeight": "bold"}),
    ], style={"backgroundColor": SURFACE, "borderRadius": "10px",
              "padding": "14px 20px", "border": f"1px solid {BORDER}",
              "textAlign": "center", "minWidth": "110px"})

def pill(text, color):
    return html.Span(text, style={
        "backgroundColor": color + "33", "color": color,
        "border": f"1px solid {color}66", "padding": "2px 8px",
        "borderRadius": "6px", "fontSize": "10px", "fontWeight": "bold",
    })

# ══════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════

def run():
    print("Loading datasets...")
    DATASETS = build_datasets()
    print(f"Loaded {len(DATASETS)} datasets")

    # Build DataFrame for summary
    META = pd.DataFrame([{k: v for k, v in d.items() if k not in ('X','y')}
                          for d in DATASETS])
    DS_BY_NAME = {d['name']: d for d in DATASETS}

    app = dash.Dash(__name__, suppress_callback_exceptions=True)

    tab_s = {"backgroundColor": SURFACE, "color": MUTED,
              "border": f"1px solid {BORDER}", "borderRadius": "8px 8px 0 0",
              "padding": "10px 20px", "fontFamily": "monospace", "fontSize": "13px"}
    tab_a = {**tab_s, "backgroundColor": CARD, "color": C_BLUE,
              "borderBottom": f"2px solid {C_BLUE}"}

    ds_options = [{"label": d['name'], "value": d['name']} for d in DATASETS]
    cond_options = [{"label": f"C{c} — {COND_NAMES[c]}", "value": c}
                    for c in sorted(COND_NAMES)]

    app.layout = html.Div(
        style={"backgroundColor": BG, "minHeight": "100vh",
               "fontFamily": "monospace", "padding": "24px"},
        children=[
            # Header
            html.Div([
                html.H1("IK Partitioning Study — EDA",
                        style={"color": "white", "margin": "0",
                               "fontSize": "20px", "letterSpacing": "3px"}),
                html.P(
                    "Exploratory Data Analysis · 41 Datasets · 7 Conditions · "
                    "4 Partitioning Mechanisms",
                    style={"color": MUTED, "margin": "6px 0 0 0", "fontSize": "11px"}),
            ], style={"marginBottom": "20px",
                       "borderBottom": f"1px solid {BORDER}",
                       "paddingBottom": "16px"}),

            # Tabs
            dcc.Tabs(id="tabs", value="overview", children=[
                dcc.Tab(label="Overview",     value="overview",     style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Scatter",      value="scatter",      style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Distributions",value="distributions",style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Conditions",   value="conditions",   style=tab_s, selected_style=tab_a),
                dcc.Tab(label="Coverage",     value="coverage",     style=tab_s, selected_style=tab_a),
            ]),

            html.Div(id="tab-content",
                     style={"backgroundColor": CARD, "borderRadius": "0 12px 12px 12px",
                             "border": f"1px solid {BORDER}", "padding": "24px",
                             "minHeight": "700px"}),

            # Hidden stores
            dcc.Store(id="meta-store", data=META.to_dict('records')),
        ]
    )

    # ── TAB ROUTER ─────────────────────────────────────────────────────────
    @callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab):

        # ════════════════════════════════════════════════════════════════
        # TAB 1: OVERVIEW
        # ════════════════════════════════════════════════════════════════
        if tab == "overview":
            total = len(DATASETS)
            n_c   = int((META['task']=='C').sum())
            n_ad  = int((META['task']=='AD').sum())
            n_real  = int(META['source'].isin(['sklearn','UCI','ADBench']).sum())
            n_synth = int((META['source']=='sklearn_gen').sum())

            # Condition bar chart
            cond_c  = META[META['task']=='C']['condition'].value_counts().sort_index()
            cond_ad = META[META['task']=='AD']['condition'].value_counts().sort_index()
            conds = list(range(1,8))

            cov_fig = go.Figure()
            cov_fig.add_trace(go.Bar(
                x=[f"C{c}" for c in conds],
                y=[cond_c.get(c,0) for c in conds],
                name="Clustering", marker_color=C_GREEN, opacity=0.85,
                text=[cond_c.get(c,0) for c in conds], textposition="auto",
            ))
            cov_fig.add_trace(go.Bar(
                x=[f"C{c}" for c in conds],
                y=[cond_ad.get(c,0) for c in conds],
                name="Anomaly Detection", marker_color=C_ORANGE, opacity=0.85,
                text=[cond_ad.get(c,0) for c in conds], textposition="auto",
            ))
            cov_fig.add_hline(y=2, line_dash="dash", line_color=C_PINK,
                              annotation_text="Min required (2)",
                              annotation_font_color=C_PINK)
            cov_fig.update_layout(
                **base_layout, barmode="stack",
                title="Datasets per condition",
                xaxis=dict(gridcolor=GRID,
                           ticktext=[f"C{c}<br>{COND_NAMES[c]}" for c in conds],
                           tickvals=[f"C{c}" for c in conds]),
                yaxis=dict(gridcolor=GRID, title="Count"),
                height=350,
            )

            # Source pie
            src_counts = META['source'].value_counts()
            src_pie = go.Figure(go.Pie(
                labels=src_counts.index.tolist(),
                values=src_counts.values.tolist(),
                marker_colors=[SRC_COLORS.get(s,'#888') for s in src_counts.index],
                hole=0.5, textinfo="label+percent",
                textfont=dict(size=11),
            ))
            src_pie.update_layout(**{k:v for k,v in base_layout.items() if k!='margin'},
                                   margin=dict(l=10,r=10,t=40,b=10),
                                   title="Source distribution", height=300)

            # Shape distribution
            shape_counts = META['shape'].value_counts()
            shape_bar = go.Figure(go.Bar(
                x=shape_counts.index.tolist(),
                y=shape_counts.values.tolist(),
                marker_color=[SHAPE_COLORS.get(s,'#888') for s in shape_counts.index],
                opacity=0.85,
                text=shape_counts.values.tolist(), textposition="outside",
            ))
            shape_bar.update_layout(
                **base_layout,
                title="Cluster shape coverage",
                xaxis=dict(gridcolor=GRID),
                yaxis=dict(gridcolor=GRID, title="Count"),
                height=300, showlegend=False,
            )

            # Task balance
            task_pie = go.Figure(go.Pie(
                labels=["Clustering (C)", "Anomaly Detection (AD)"],
                values=[n_c, n_ad],
                marker_colors=[C_GREEN, C_ORANGE],
                hole=0.5, textinfo="label+percent+value",
            ))
            task_pie.update_layout(**{k:v for k,v in base_layout.items() if k!='margin'},
                                    margin=dict(l=10,r=10,t=40,b=10),
                                    title="Task balance", height=300)

            return html.Div([
                # Stats row
                html.Div([
                    big_stat("Total datasets", total,  "white"),
                    big_stat("Clustering",     n_c,    C_GREEN),
                    big_stat("Anomaly detect.", n_ad,   C_ORANGE),
                    big_stat("Real datasets",   n_real, C_BLUE),
                    big_stat("Synthetic",       n_synth,C_PURPLE),
                    big_stat("Conditions",      7,      C_YELLOW),
                ], style={"display":"flex","gap":"10px","flexWrap":"wrap",
                           "marginBottom":"20px"}),

                # Coverage chart (full width)
                card([dcc.Graph(figure=cov_fig)], mb=12),

                # 3 charts row
                html.Div([
                    html.Div(card([dcc.Graph(figure=task_pie)], mb=0), style={"flex":"1"}),
                    html.Div(card([dcc.Graph(figure=src_pie)],  mb=0), style={"flex":"1"}),
                    html.Div(card([dcc.Graph(figure=shape_bar)],mb=0), style={"flex":"1"}),
                ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),

                # Key observations
                card([
                    sec("Key observations for the study", C_GREEN),
                    html.Ul([
                        html.Li(
                            "Conditions 3 and 4 have 0 real datasets with "
                            "known crescent/nested structure — synthetic datasets "
                            "are essential here (cannot be avoided).",
                            style={"color": TEXT, "marginBottom":"6px","fontSize":"12px"}),
                        html.Li(
                            "Condition 4 (nested/concentric) is most underrepresented. "
                            "This is a known limitation — nested shapes are rare in real data.",
                            style={"color": TEXT, "marginBottom":"6px","fontSize":"12px"}),
                        html.Li(
                            "Condition 6 has the most datasets (8) because "
                            "high-dimensional data is common in real benchmarks (ODDS, UCI).",
                            style={"color": TEXT, "marginBottom":"6px","fontSize":"12px"}),
                        html.Li(
                            "All 7 conditions have ≥ 2 datasets — minimum coverage met. "
                            "Most have ≥ 4 datasets for robust comparison.",
                            style={"color": TEXT, "fontSize":"12px"}),
                    ]),
                ]),
            ])

        # ════════════════════════════════════════════════════════════════
        # TAB 2: SCATTER
        # ════════════════════════════════════════════════════════════════
        elif tab == "scatter":
            return html.Div([
                card([
                    sec("Dataset scatter plot (2D / PCA projection)"),
                    html.Div([
                        html.Span("Dataset: ", style={"color":MUTED,"marginRight":"10px","fontSize":"12px"}),
                        dcc.Dropdown(
                            id="scatter-ds",
                            options=ds_options,
                            value=DATASETS[0]['name'],
                            style={"width":"320px","backgroundColor":SURFACE,"color":"#111",
                                   "border":f"1px solid {BORDER}"},
                        ),
                        html.Span("  or filter by condition: ",
                                  style={"color":MUTED,"marginLeft":"20px","marginRight":"10px","fontSize":"12px"}),
                        dcc.Dropdown(
                            id="scatter-cond",
                            options=cond_options,
                            value=None,
                            placeholder="All conditions",
                            style={"width":"260px","backgroundColor":SURFACE,"color":"#111",
                                   "border":f"1px solid {BORDER}"},
                        ),
                    ], style={"display":"flex","alignItems":"center","marginBottom":"16px"}),
                    dcc.Graph(id="scatter-graph", style={"height":"500px"}),
                    html.Div(id="scatter-info",
                             style={"color":MUTED,"fontSize":"11px","marginTop":"8px"}),
                ]),

                card([
                    sec("All 2D datasets at a glance"),
                    dcc.Graph(id="scatter-grid"),
                ]),
            ])

        # ════════════════════════════════════════════════════════════════
        # TAB 3: DISTRIBUTIONS
        # ════════════════════════════════════════════════════════════════
        elif tab == "distributions":
            # n vs features bubble chart
            bubble = go.Figure()
            for src, color in SRC_COLORS.items():
                sub = META[META['source']==src]
                bubble.add_trace(go.Scatter(
                    x=sub['n'], y=sub['features'],
                    mode='markers+text',
                    name=src,
                    marker=dict(
                        size=10, color=color, opacity=0.85,
                        line=dict(color='white', width=0.5)
                    ),
                    text=sub['name'],
                    textposition="top center",
                    textfont=dict(size=7, color=MUTED),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "n=%{x:,}<br>features=%{y}<extra></extra>"
                    ),
                ))
            bubble.update_layout(
                **base_layout,
                title="n (samples) vs features — by source",
                xaxis=dict(type="log", gridcolor=GRID, title="n (log scale)"),
                yaxis=dict(type="log", gridcolor=GRID, title="features (log scale)"),
                height=420,
            )

            # Anomaly rate bar
            ad_meta = META[META['task']=='AD'].dropna(subset=['anom_rate'])
            ad_sorted = ad_meta.sort_values('anom_rate')
            anom_bar = go.Figure(go.Bar(
                x=ad_sorted['anom_rate'],
                y=ad_sorted['name'],
                orientation='h',
                marker_color=[COND_COLORS.get(int(c),'#888')
                              for c in ad_sorted['condition']],
                opacity=0.85,
                text=[f"{v:.1f}%" for v in ad_sorted['anom_rate']],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Anomaly rate: %{x:.1f}%<extra></extra>",
            ))
            anom_layout = {k: v for k, v in base_layout.items() if k != 'margin'}
            anom_bar.update_layout(
                **anom_layout,
                title="Anomaly rate per AD dataset (coloured by condition)",
                xaxis=dict(gridcolor=GRID, title="Anomaly rate (%)"),
                yaxis=dict(gridcolor=GRID),
                height=max(300, len(ad_meta) * 28),
                margin=dict(l=150, r=60, t=50, b=50),
            )
            # Add reference line at 5%
            anom_bar.add_vline(x=5, line_dash="dash", line_color=C_PINK,
                               annotation_text="5%", annotation_font_color=C_PINK)

            # n histogram
            n_hist = go.Figure()
            for task, color, name in [('C',C_GREEN,'Clustering'),('AD',C_ORANGE,'Anomaly')]:
                vals = META[META['task']==task]['n']
                n_hist.add_trace(go.Histogram(
                    x=vals, name=name, marker_color=color,
                    opacity=0.7, nbinsx=20,
                ))
            n_hist.update_layout(
                **base_layout,
                barmode='overlay',
                title="Sample count distribution",
                xaxis=dict(type='log', gridcolor=GRID, title="n (log scale)"),
                yaxis=dict(gridcolor=GRID, title="Count"),
                height=280,
            )

            # features histogram
            f_hist = go.Figure()
            f_hist.add_trace(go.Histogram(
                x=META['features'], marker_color=C_PURPLE,
                opacity=0.85, nbinsx=20,
            ))
            f_hist.update_layout(
                **base_layout,
                title="Feature count distribution",
                xaxis=dict(type='log', gridcolor=GRID, title="features (log scale)"),
                yaxis=dict(gridcolor=GRID, title="Count"),
                height=280, showlegend=False,
            )

            return html.Div([
                card([dcc.Graph(figure=bubble)], mb=12),
                html.Div([
                    html.Div(card([dcc.Graph(figure=n_hist)], mb=0),  style={"flex":"1"}),
                    html.Div(card([dcc.Graph(figure=f_hist)], mb=0),  style={"flex":"1"}),
                ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
                card([dcc.Graph(figure=anom_bar)]),
            ])

        # ════════════════════════════════════════════════════════════════
        # TAB 4: CONDITIONS
        # ════════════════════════════════════════════════════════════════
        elif tab == "conditions":
            return html.Div([
                card([
                    sec("Select condition"),
                    dcc.Dropdown(
                        id="cond-select",
                        options=cond_options,
                        value=1,
                        style={"width":"380px","backgroundColor":SURFACE,
                               "color":"#111","border":f"1px solid {BORDER}"},
                    ),
                ], mb=12),
                html.Div(id="cond-detail"),
            ])

        # ════════════════════════════════════════════════════════════════
        # TAB 5: COVERAGE CHECK
        # ════════════════════════════════════════════════════════════════
        elif tab == "coverage":
            dim_order  = {"low":0,"mid":1,"high":2,"vhigh":3}
            size_order = {"small":0,"medium":1,"large":2}

            # Heatmap: condition × dim_level
            dim_levels = ["low","mid","high","vhigh"]
            z_dim = [[int((META[(META['condition']==c)&(META['dim_level']==d)].shape[0]))
                      for d in dim_levels] for c in range(1,8)]
            hm_dim = go.Figure(go.Heatmap(
                z=z_dim,
                x=dim_levels, y=[f"C{c} {COND_NAMES[c]}" for c in range(1,8)],
                colorscale=[[0,"#1a1a35"],[0.3,"#1D5F99"],[1,"#6af0f7"]],
                text=[[str(v) for v in row] for row in z_dim],
                texttemplate="%{text}", textfont=dict(size=14),
                hovertemplate="Cond %{y}<br>Dim: %{x}<br>Count: %{z}<extra></extra>",
            ))
            hm_dim.update_layout(
                **base_layout,
                title="Coverage: condition × dimensionality",
                height=320,
                xaxis=dict(gridcolor=GRID),
                yaxis=dict(gridcolor=GRID),
            )

            # Heatmap: condition × size_level
            size_levels = ["small","medium","large"]
            z_size = [[int(META[(META['condition']==c)&(META['size_level']==s)].shape[0])
                       for s in size_levels] for c in range(1,8)]
            hm_size = go.Figure(go.Heatmap(
                z=z_size,
                x=size_levels, y=[f"C{c} {COND_NAMES[c]}" for c in range(1,8)],
                colorscale=[[0,"#1a1a35"],[0.3,"#5a3090"],[1,"#9b7ff7"]],
                text=[[str(v) for v in row] for row in z_size],
                texttemplate="%{text}", textfont=dict(size=14),
            ))
            hm_size.update_layout(
                **base_layout,
                title="Coverage: condition × dataset size",
                height=320,
            )

            # Source × task heatmap
            sources = ["sklearn","sklearn_gen","UCI","ADBench"]
            tasks   = ["C","AD"]
            z_src = [[int(META[(META['source']==src)&(META['task']==t)].shape[0])
                      for src in sources] for t in tasks]
            hm_src = go.Figure(go.Heatmap(
                z=z_src,
                x=sources, y=tasks,
                colorscale=[[0,"#1a1a35"],[0.5,"#3d7a2e"],[1,"#6af7a0"]],
                text=[[str(v) for v in row] for row in z_src],
                texttemplate="%{text}", textfont=dict(size=16),
            ))
            hm_src.update_layout(
                **base_layout,
                title="Source × task coverage",
                height=220,
            )

            # Gap analysis
            gaps = []
            for c in range(1,8):
                sub = META[META['condition']==c]
                n_c_  = int((sub['task']=='C').sum())
                n_ad_ = int((sub['task']=='AD').sum())
                if n_c_ == 0:
                    gaps.append(f"C{c} ({COND_NAMES[c]}): no Clustering datasets")
                if n_ad_ == 0:
                    gaps.append(f"C{c} ({COND_NAMES[c]}): no AD datasets")
                if len(sub) < 2:
                    gaps.append(f"C{c} ({COND_NAMES[c]}): fewer than 2 datasets total")

            return html.Div([
                html.Div([
                    html.Div(card([dcc.Graph(figure=hm_dim)],  mb=0), style={"flex":"1"}),
                    html.Div(card([dcc.Graph(figure=hm_size)], mb=0), style={"flex":"1"}),
                ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),
                card([dcc.Graph(figure=hm_src)], mb=12),
                card([
                    sec("Gap analysis", C_ORANGE if gaps else C_GREEN),
                    html.Div([
                        html.P("All conditions covered ✓",
                               style={"color":C_GREEN,"fontSize":"13px"})
                    ] if not gaps else [
                        html.Li(g, style={"color":C_ORANGE,"fontSize":"12px","marginBottom":"4px"})
                        for g in gaps
                    ]),
                ]),
            ])

        return html.P("Select a tab.", style={"color": MUTED})

    # ── SCATTER CALLBACKS ──────────────────────────────────────────────────
    @callback(
        Output("scatter-graph", "figure"),
        Output("scatter-info",  "children"),
        Input("scatter-ds",   "value"),
    )
    def update_scatter(ds_name):
        if not ds_name or ds_name not in DS_BY_NAME:
            return go.Figure(), ""
        d = DS_BY_NAME[ds_name]
        X, y = d['X'], d['y']

        if X.shape[1] > 2:
            X2 = PCA(n_components=2, random_state=42).fit_transform(X)
            proj = f"PCA projection (from {X.shape[1]} features)"
        else:
            X2 = X
            proj = "Original 2D"

        # Sample
        idx = np.random.RandomState(42).choice(len(X2), min(1500, len(X2)), replace=False)
        X_p, y_p = X2[idx], y[idx]

        n_cl = len(np.unique(y_p))
        if d['task'] == 'AD':
            color_map = {0: C_BLUE, 1: C_ORANGE}
            label_map = {0: "normal", 1: "anomaly"}
        else:
            palette = [C_BLUE, C_PURPLE, C_GREEN, C_ORANGE, C_YELLOW,
                       C_PINK, C_TEAL, "#aaaaff", "#ffaaaa", "#aaffaa"]
            color_map = {i: palette[i % len(palette)] for i in range(n_cl)}
            label_map = {i: f"class {i}" for i in range(n_cl)}

        fig = go.Figure()
        for cls in sorted(np.unique(y_p)):
            mask = y_p == cls
            fig.add_trace(go.Scatter(
                x=X_p[mask,0], y=X_p[mask,1],
                mode='markers',
                name=label_map.get(int(cls), str(cls)),
                marker=dict(color=color_map.get(int(cls),'#888'),
                            size=5, opacity=0.7,
                            line=dict(color='white', width=0.3)),
            ))
        fig.update_layout(
            **base_layout,
            title=f"{ds_name}  —  {proj}",
            xaxis=dict(gridcolor=GRID, title="Component 1"),
            yaxis=dict(gridcolor=GRID, title="Component 2"),
            height=480,
        )
        info = (f"Task: {d['task']}  |  n={d['n']:,}  |  "
                f"features={d['features']}  |  "
                f"shape={d['shape']}  |  "
                f"condition={d['condition']} — {COND_NAMES[d['condition']]}  |  "
                f"source={d['source']}")
        return fig, info

    @callback(Output("scatter-grid", "figure"), Input("scatter-cond", "value"))
    def update_scatter_grid(cond_filter):
        if cond_filter:
            subset = [d for d in DATASETS if d['condition'] == cond_filter]
        else:
            subset = [d for d in DATASETS if d['features'] <= 3]

        if not subset:
            return go.Figure()

        ncols = 4
        nrows = max(1, (len(subset) + ncols - 1) // ncols)
        titles = [d['name'] for d in subset]

        fig = make_subplots(rows=nrows, cols=ncols,
                             subplot_titles=titles)

        for i, d in enumerate(subset):
            r = i // ncols + 1
            c = i % ncols + 1
            X, y = d['X'], d['y']
            if X.shape[1] > 2:
                X2 = PCA(n_components=2, random_state=42).fit_transform(X)
            else:
                X2 = X

            idx = np.random.RandomState(42).choice(len(X2), min(400, len(X2)), replace=False)
            X_p, y_p = X2[idx], y[idx]

            if d['task'] == 'AD':
                palette = {0: C_BLUE, 1: C_ORANGE}
            else:
                pal = [C_BLUE, C_PURPLE, C_GREEN, C_ORANGE,
                       C_YELLOW, C_PINK, C_TEAL]
                palette = {j: pal[j % len(pal)] for j in range(len(np.unique(y_p)))}

            cols = [palette.get(int(yi), '#888') for yi in y_p]
            fig.add_trace(
                go.Scatter(x=X_p[:,0], y=X_p[:,1], mode='markers',
                           marker=dict(color=cols, size=4, opacity=0.7),
                           showlegend=False),
                row=r, col=c
            )

        fig.update_layout(
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            font=dict(color=TEXT, size=9, family="monospace"),
            height=nrows * 220,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        fig.update_xaxes(showticklabels=False, gridcolor=GRID, zeroline=False)
        fig.update_yaxes(showticklabels=False, gridcolor=GRID, zeroline=False)
        return fig

    # ── CONDITION DETAIL CALLBACK ──────────────────────────────────────────
    @callback(Output("cond-detail", "children"), Input("cond-select", "value"))
    def update_cond(cond):
        if cond is None:
            return html.P("Select a condition.", style={"color":MUTED})

        cond = int(cond)
        sub  = META[META['condition']==cond]
        n_c  = int((sub['task']=='C').sum())
        n_ad = int((sub['task']=='AD').sum())
        color = COND_COLORS.get(cond, C_BLUE)

        # Why this condition matters
        why_card = card([
            html.H2(f"C{cond} — {COND_NAMES[cond]}",
                    style={"color": color, "margin":"0 0 10px 0", "fontSize":"18px"}),
            html.P(COND_WHY.get(cond,""),
                   style={"color": TEXT, "fontSize":"12px", "lineHeight":"1.7",
                           "margin":"0 0 12px 0"}),
            html.Div([
                big_stat("Datasets",   len(sub), color),
                big_stat("Clustering", n_c,      C_GREEN),
                big_stat("Anomaly",    n_ad,     C_ORANGE),
                big_stat("Real",       int(sub['source'].isin(['sklearn','UCI','ADBench']).sum()), C_BLUE),
                big_stat("Synthetic",  int((sub['source']=='sklearn_gen').sum()), C_PURPLE),
            ], style={"display":"flex","gap":"10px","flexWrap":"wrap","marginTop":"10px"}),
        ], mb=12)

        # Dataset table
        th_s = {"padding":"8px 12px","color":C_PURPLE,"textAlign":"left",
                 "backgroundColor":GRID,"fontSize":"11px"}
        td_s = {"padding":"7px 12px","fontSize":"11px",
                 "borderBottom":f"1px solid {GRID}"}

        rows_html = []
        for _, row in sub.sort_values(['task','n']).iterrows():
            src_color = SRC_COLORS.get(row['source'], '#888')
            task_color = C_GREEN if row['task']=='C' else C_ORANGE
            rows_html.append(html.Tr([
                html.Td(html.Span(row['name'], style={"fontWeight":"bold","color":TEXT}), style=td_s),
                html.Td(html.Span(row['task'], style={"color":task_color,"fontWeight":"bold"}), style=td_s),
                html.Td(f"{row['n']:,}", style={**td_s,"color":TEXT}),
                html.Td(str(row['features']), style={**td_s,"color":TEXT}),
                html.Td(row['shape'], style={**td_s,"color":SHAPE_COLORS.get(row['shape'],'#888')}),
                html.Td(row['density'], style={**td_s,"color":TEXT}),
                html.Td(row['dim_level'], style={**td_s,"color":TEXT}),
                html.Td(row['size_level'], style={**td_s,"color":TEXT}),
                html.Td(html.Span(row['source'],
                                  style={"color":src_color,"fontWeight":"bold",
                                          "fontSize":"10px"}), style=td_s),
                html.Td(f"{row['anom_rate']:.1f}%" if row['anom_rate'] else "—",
                        style={**td_s,"color":C_ORANGE if row['anom_rate'] else MUTED}),
            ]))

        table = card([
            sec(f"Datasets in condition {cond}"),
            html.Table([
                html.Tr([html.Th(h, style=th_s) for h in
                         ["Name","Task","n","Features","Shape","Density",
                          "Dim","Size","Source","Anom%"]])
            ] + rows_html, style={"width":"100%","borderCollapse":"collapse"}),
        ])

        # Expected results based on literature
        expected = {
            1: "All 4 partitions should perform similarly. "
               "This is your control condition.",
            2: "Expected: random hyperplane > axis-parallel (iForest). "
               "Voronoi and hypersphere may also beat iForest on diagonal clusters.",
            3: "From the survey paper: iForest AUC drops significantly. "
               "Hypersphere (iNNE) should achieve near-perfect AUC.",
            4: "Hypersphere and Voronoi expected to outperform hyperplane methods. "
               "The gap should be the largest across all conditions.",
            5: "Hypersphere (iNNE) expected to win — it adapts ball size to density. "
               "iForest produces equal-sized regions regardless of density.",
            6: "Random hyperplane expected to be most robust. "
               "Voronoi degrades due to high-dimensional Euclidean distance issues.",
            7: "iForest expected to be fastest (O(n log n) tree construction). "
               "Focus on runtime, not just accuracy.",
        }

        exp_card = card([
            sec("Expected results (based on theory + literature)", C_YELLOW),
            html.P(expected.get(cond, ""),
                   style={"color": TEXT, "fontSize":"12px", "lineHeight":"1.7"}),
        ])

        return html.Div([why_card, table, exp_card])

    print(f"\n  EDA Dashboard → http://127.0.0.1:8053\n")
    app.run(debug=False, use_reloader=False, port=8053)


if __name__ == "__main__":
    run()