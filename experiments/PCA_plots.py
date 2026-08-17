#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import argparse
from collections import Counter
import plotly.express as px
import copy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_and_filter, get_keep_indices


def compute_PCA(input_type, model, d, min_count=40):
    embeddings, labels, source = load_and_filter(input_type, model, min_count=min_count)
    pca = PCA(n_components=d)
    projections = pca.fit_transform(embeddings)

    results = {
        'pca': pca,
        'projections': projections,
        'embeddings': embeddings,
        'labels': labels,
        'source': source,
    }

    return results

def plot_pca_results(results, title=None, marker_size=5, opacity=1,
                     width=None, height=550):
    """
    Plot 2D PCA projections. `results` can be either:
      - a dict mapping (input_type, model) -> result dict (multiple panels), or
      - a single result dict with keys 'projections', 'labels', 'pca' (one panel).
    """
    # Accept a single result dict by wrapping it
    if 'projections' in results:
        results = {'_single': results}
        is_single = True
    else:
        is_single = False

    keys = list(results.keys())
    n = len(keys)
    if width is None:
        width = 700 if is_single else 1200

    all_labels = np.concatenate([np.asarray(results[k]['labels']) for k in keys])
    unique_labels = sorted(set(all_labels.tolist()))
    palette = px.colors.qualitative.Set3
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    subplot_titles = None if is_single else [f"{inp} — {mdl}" for (inp, mdl) in keys]
    fig = make_subplots(rows=1, cols=n, subplot_titles=subplot_titles,
                        horizontal_spacing=0.08)

    seen = set()
    for col, key in enumerate(keys, start=1):
        proj = np.asarray(results[key]['projections'])
        labels = np.asarray(results[key]['labels'])
        pca = results[key]['pca']

        for lab in unique_labels:
            mask = labels == lab
            if not mask.any():
                continue
            show = lab not in seen
            seen.add(lab)
            fig.add_trace(
                go.Scatter(
                    x=proj[mask, 0], y=proj[mask, 1],
                    mode='markers',
                    name=str(lab),
                    legendgroup=str(lab),
                    showlegend=show,
                    marker=dict(size=marker_size, color=color_map[lab],
                                opacity=opacity, line=dict(width=0)),
                    hovertemplate=f"<b>{lab}</b><br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>",
                ),
                row=1, col=col,
            )

        ev = pca.explained_variance_ratio_
        fig.update_xaxes(title_text=f"PC1 ({ev[0]*100:.1f}%)", row=1, col=col,
                         zeroline=True, zerolinecolor='lightgrey')
        fig.update_yaxes(title_text=f"PC2 ({ev[1]*100:.1f}%)", row=1, col=col,
                         zeroline=True, zerolinecolor='lightgrey',
                         scaleanchor=f"x{col}", scaleratio=1)

    fig.update_layout(
        title=title or "PCA projections",
        width=width, height=height,
        template="plotly_white",
        legend=dict(title="Label", itemsizing='constant',
                    bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=160, t=70, b=60),
    )
    return fig


def plot_pca_results_3d(results, title=None, marker_size=1.75, opacity=1,
                        width=None, height=650):
    """
    Plot 3D PCA projections. `results` can be either:
      - a dict mapping (input_type, model) -> result dict (multiple panels), or
      - a single result dict with keys 'projections', 'labels', 'pca' (one panel).
    """
    if 'projections' in results:
        results = {'_single': results}
        is_single = True
    else:
        is_single = False

    keys = list(results.keys())
    n = len(keys)
    if width is None:
        width = 750 if is_single else 1300

    all_labels = np.concatenate([np.asarray(results[k]['labels']) for k in keys])
    unique_labels = sorted(set(all_labels.tolist()))
    palette = px.colors.qualitative.Set3
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(unique_labels)}

    subplot_titles = None if is_single else [f"{inp} — {mdl}" for (inp, mdl) in keys]
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'scene'}] * n],
        horizontal_spacing=0.04,
    )

    seen = set()
    for col, key in enumerate(keys, start=1):
        proj = np.asarray(results[key]['projections'])
        labels = np.asarray(results[key]['labels'])
        pca = results[key]['pca']
        if proj.shape[1] < 3:
            raise ValueError(
                f"Need at least 3 PCA components for 3D plot, got {proj.shape[1]} "
                f"for {key}. Re-run compute_PCA with d=3."
            )

        for lab in unique_labels:
            mask = labels == lab
            if not mask.any():
                continue
            show = lab not in seen
            seen.add(lab)
            fig.add_trace(
                go.Scatter3d(
                    x=proj[mask, 0], y=proj[mask, 1], z=proj[mask, 2],
                    mode='markers',
                    name=str(lab),
                    legendgroup=str(lab),
                    showlegend=show,
                    marker=dict(size=marker_size, color=color_map[lab],
                                opacity=opacity, line=dict(width=0)),
                    hovertemplate=(
                        f"<b>{lab}</b><br>"
                        "PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

        ev = pca.explained_variance_ratio_
        scene_id = 'scene' if col == 1 else f'scene{col}'
        fig.layout[scene_id].update(
            # xaxis_title=f"PC1 ({ev[0]*100:.1f}%)",
            # yaxis_title=f"PC2 ({ev[1]*100:.1f}%)",
            # zaxis_title=f"PC3 ({ev[2]*100:.1f}%)",
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            aspectmode='cube',
        )

    fig.update_layout(
        # title=title or "PCA projections (3D)",
        width=width, height=height,
        template="plotly_white",
        legend=dict(title="Label", itemsizing='constant',
                    bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=20, r=160, t=70, b=20),
    )
    return fig
#%% 
# compute PCA fo SBERT B and ft_SBERT B
results = {}
models = ["SBERT", "ft_SBERT"]
models = ["mean_DEBERTA", "ft_mean_DEBERTA"]
input_type = "B"

# usage — you'll need d=3 in compute_PCA
results = {}
for model in models:
    results[(input_type, model)] = compute_PCA(input_type, model, d=3, min_count=40)


fig1 = plot_pca_results_3d(results[(input_type, models[0])], opacity=1)
fig1.update_layout(width=None, height=None, autosize=True)
fig1.write_html(
    f"{models[0]}_B.html",
    full_html=True,
    include_plotlyjs="cdn",
    default_width="100%",
    default_height="100vh",
)

fig2 = plot_pca_results_3d(results[(input_type, models[1])], title=f"PCA 3D — input {input_type}")
fig2.update_layout(width=None, height=None, autosize=True)
fig2.write_html(
    f"{models[1]}_B.html",
    full_html=True,
    include_plotlyjs="cdn",
    default_width="100%",
    default_height="100vh",
)
# %%
def preview(fig, eye=(1.25, 1.25, 1.25), up=(0, 0, 1), center=(0, 0, 0)):
    """Show fig with a given camera. Returns the modified copy so you can save it."""
    f = go.Figure(fig)  # shallow copy is fine here
    f.update_layout(scene_camera=dict(
        eye=dict(x=eye[0], y=eye[1], z=eye[2]),
        up=dict(x=up[0], y=up[1], z=up[2]),
        center=dict(x=center[0], y=center[1], z=center[2]),
    ))
    f.show()
    return f
# %%
fig1.update_layout(
    # title=title or "PCA projections (3D)",
    # width=width, height=height,
    template="plotly_white",
    legend=dict(
        title=None,                # the "Label" title takes space; drop it
        orientation='h',           # horizontal row of entries
        yanchor='top', y=-0.1,    # just below the plot area
        xanchor='center', x=0.5,   # centered horizontally
        font=dict(size=20),        # smaller text
        itemsizing='constant',
        itemwidth=30,              # tighter spacing between entries (min 30)
        bgcolor='rgba(255,255,255,0.8)',
    ),
    margin=dict(l=20, r=20, t=70, b=80),   # extra bottom room, less on the right
)
#%%
# the 3 star manifold 
preview(fig1, eye=(0.35, 1.6, 1.8), up=(0, 0, 1), center=(-0.2, -0.2, -0.2))
camera = dict(
    eye=dict(x=0.35, y=1.6, z=1.8),
    up=dict(x=0, y=0, z=1),
    center=dict(x=-0.2, y=-0.2, z=-0.2)
)

fig1.update_layout(scene_camera=camera)

fig1.write_image(f"{models[0]}_B_view1.pdf")

# %%
# accuntancy cluster 
preview(fig1, eye=(-2, 0.5, 0.5), up=(0, 0, 1), center=(0, 0, 0))
camera = dict(
    eye=dict(x=-2, y=0.5, z=0.5),
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0)
)
fig1.update_layout(scene_camera=camera)
fig1.write_image(f"{models[0]}_B_view2.pdf")
#%%
# accunacnty cluster ft_SBERT
preview(fig2, eye=(-2, 0.5, 0.5), up=(0, 0, 1), center=(0, 0, 0))
camera = dict(
    eye=dict(x=-2, y=0.5, z=0.5),
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0)

)
fig2.update_layout(scene_camera=camera, 
                       legend=dict(
        title=None,                # the "Label" title takes space; drop it
        orientation='h',           # horizontal row of entries
        yanchor='top', y=0,    # just below the plot area
        xanchor='center', x=0.5,   # centered horizontally
        font=dict(size=15),        # smaller text
        itemsizing='constant',
        itemwidth=30,              # tighter spacing between entries (min 30)
        bgcolor='rgba(255,255,255,0.8)',
    ),  margin=dict(l=20, r=20, t=0, b=60))
fig2.write_image(f"{models[1]}_B_view2.pdf")
#%%
preview(fig2, eye=(1.6, -0.35, 1.8), up=(0, 0, 1), center=(-0.2, -0.2, -0.2))
camera = dict(
    eye=dict(x=1.6, y=-0.35, z=1.8),
    up=dict(x=0, y=0, z=1),
    center=dict(x=-0.2, y=-0.2, z=-0.2)
)

fig2.update_layout(scene_camera=camera)

fig2.write_image(f"{models[1]}_B_view1.pdf")

# %%
def side_by_side_3d_views(fig, cameras, width=1100, height=550, h_spacing=0.02):
    """Render the same 3D figure from multiple camera angles, one shared legend."""
    n = len(cameras)
    new_fig = make_subplots(
        rows=1, cols=n,
        specs=[[{'type': 'scene'}] * n],
        horizontal_spacing=h_spacing,
    )

    for col in range(1, n + 1):
        for trace in fig.data:
            t = copy.deepcopy(trace)
            t.showlegend = (col == 1) and bool(trace.showlegend)
            new_fig.add_trace(t, row=1, col=col)

    orig = fig.layout.scene
    get_title = lambda ax: ax.title.text if ax.title and ax.title.text else None
    for col, camera in enumerate(cameras, start=1):
        scene_id = 'scene' if col == 1 else f'scene{col}'
        new_fig.layout[scene_id].update(
            xaxis_title=get_title(orig.xaxis),
            yaxis_title=get_title(orig.yaxis),
            zaxis_title=get_title(orig.zaxis),
            aspectmode='cube',
            camera=camera,
        )

    new_fig.update_layout(
        template="plotly_white",
        width=width, height=height,
        margin=dict(l=0, r=0, t=1, b=70),
        legend=dict(
            title=None,
            orientation='h',
            yanchor='top', y=-0.02,
            xanchor='center', x=0.5,
            font=dict(size=18),
            itemsizing='constant',
            itemwidth=30,
            bgcolor='rgba(255,255,255,0.8)',
        ),
    )
    return new_fig

#%%

cam1 = dict(eye=dict(x=1.5, y=0.5, z=0),
            up=dict(x=0, y=0, z=1),
            center=dict(x=-0.2, y=-0.2, z=-0.2))
cam2 = dict(eye=dict(x=1, y=-1.4, z=0.7),
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0))

combined = side_by_side_3d_views(fig1, [cam2, cam1])
combined.show()
#%%
combined.write_image(f"{models[0]}_B_two_views.pdf")
# %%
cam1 = dict(
    eye=dict(x=1.2, y=1.2, z=0.2),
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0)
)
cam2 = dict(
    eye=dict(x=0.5, y=0.75, z=1.8),
    up=dict(x=0, y=0, z=1),
    center=dict(x=-0.2, y=-0.2, z=-0.2)
)

combined = side_by_side_3d_views(fig2, [cam1, cam2])
combined.show()
#%%
combined.write_image(f"{models[1]}_B_two_views.pdf")
# %%
