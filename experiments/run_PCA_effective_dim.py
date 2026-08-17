
from html import parser
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import argparse
from collections import Counter
import plotly.express as px

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import INPUT_TYPES, MODELS
from utils import load_and_filter, get_keep_indices


def compute_effective_dimensions(input_types, models, d, threshold, min_count=1):
    cache = {}
    results = pd.DataFrame(index=models, columns=input_types)

    for input_type in input_types:
        for model in models:
            embeddings, labels, source = load_and_filter(input_type, model, min_count=min_count)
            pca = PCA(n_components=d)
            projections = pca.fit_transform(embeddings)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            effective_dim = np.argmax(cumulative >= threshold) + 1
            results.loc[model, input_type] = effective_dim
            print(f'{model} - {input_type}: Effective Dimension = {effective_dim}')

            cache[(input_type, model)] = {
                'pca': pca,
                'projections': projections,
                'embeddings': embeddings,
                'labels': labels,
                'source': source,
                'effective_dim': effective_dim,
            }

    print(results)
    return results, cache


def plots3d(cache):
    colors = px.colors.qualitative.Plotly

    for (input_type, model), entry in cache.items():
        projections = entry['projections'][:, :3]
        labels_arr = np.array(entry['labels'])

        counts = Counter(labels_arr)
        top9 = [label for label, _ in counts.most_common(9)]

        fig = go.Figure()

        # Plot top 9 individually
        for i, label in enumerate(top9):
            mask = labels_arr == label

            fig.add_trace(go.Scatter3d(
                x=projections[mask, 0],
                y=projections[mask, 1],
                z=projections[mask, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    opacity=0.6,
                    color=colors[i]
                ),
                name=label
            ))

        # Plot all remaining labels as "Other"
        other_mask = ~np.isin(labels_arr, top9)

        fig.add_trace(go.Scatter3d(
            x=projections[other_mask, 0],
            y=projections[other_mask, 1],
            z=projections[other_mask, 2],
            mode='markers',
            marker=dict(
                size=3,
                opacity=0.6,
                color=colors[9]
            ),
            name='Other'
        ))

        fig.update_layout(
            title=f'{model} - {input_type}',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            )
        )

        filename = f"PCA_3d_{model}_{input_type}.html"

        fig.write_html(filename)

        print(f"Saved: {filename}")


def pc1_norm_correlations(cache, input_types, models):
    for type in input_types:
        for model in models:
            entry = cache[(type, model)]
            pc1_values = entry['projections'][:, 0]
            embeddings = entry['embeddings']
            norms = np.linalg.norm(embeddings, axis=1)
            corr = np.corrcoef(pc1_values, norms)[0, 1]
            print(f'{model} - {type}: PC1 vs norm r={corr:.3f}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Skip 3D plots")
    parser.add_argument("--min_count", type=int, default=1,
                        help="Drop classes with fewer than this many samples (1 = no filtering)")
    parser.add_argument("--d", type=int, default=380, help="Number of PCA components")
    parser.add_argument("--threshold", type=float, default=0.8, help="Variance threshold for effective dimension")
    
    args = parser.parse_args()
    input_types = INPUT_TYPES
    models = MODELS


    results, cache = compute_effective_dimensions(
        input_types, models, args.d, args.threshold, min_count=args.min_count
    )
    
    results.to_csv("effective_dimensions.csv")

    if not args.no_plot:
        plots3d(cache)

    pc1_norm_correlations(cache, input_types, models)


if __name__ == "__main__":
    main()
