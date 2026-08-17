
import argparse

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import INPUT_TYPES, MODELS
from utils import load_and_filter, get_keep_indices
from cosine_similarity import compute_class_similarities, summarize_cosine_similarities, plot_similarity_distribution




def compute_global_similarities(input_types, models, min_count):
    """Average pairwise cosine similarity per (model, input_type)."""
    results = pd.DataFrame(index=models, columns=input_types)
    for input_type in input_types:
        for model in models:
            embeddings, _, _ = load_and_filter(input_type, model, min_count)
            sim_matrix = cosine_similarity(embeddings)
            triu = np.triu_indices(len(embeddings), k=1)
            results.loc[model, input_type] = sim_matrix[triu].mean()
    return results
 
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true", help="Skip distribution plots")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum label count for filtering (default: 1)")
    args = parser.parse_args()

    # Summary of filtering
    keep_indices = get_keep_indices(min_count=args.min_count)
    total = sum(len(idx) for idx in keep_indices.values())
    print(f"Keeping {total} samples after filtering labels with less than {args.min_count} samples")
 
    # Global average similarities
    results = compute_global_similarities(INPUT_TYPES, MODELS, min_count=args.min_count)
    print(f"Global average cosine similarities:\n{results}")
    results.to_csv("average_cosine_similarities.csv")
 
    # Within / between class (uncentered)
    df_within, df_between = compute_class_similarities(centering=False, min_count=args.min_count, save_csv=False)
    df_within.to_csv("within_class_cosine_similarities.csv", index=False)
    df_between.to_csv("between_class_cosine_similarities.csv", index=False)
 
    avg = summarize_cosine_similarities(df_within, df_between, save_csv=False)
    print(f"Average within & between (uncentered):\n{avg}")
 
    # Within / between class (centered)
    df_within_c, df_between_c = compute_class_similarities(centering=True, min_count=args.min_count,     save_csv=False)
    avg_c = summarize_cosine_similarities(df_within_c, df_between_c, save_csv=False)
    print(f"Average within & between (centered):\n{avg_c}")
 
    # Distribution plots
    if not args.no_plot:
        plot_similarity_distribution("SBERT", "B", min_count=args.min_count, class_labels=None)
        plot_similarity_distribution("SBERT", "B", class_labels=["Purchases"], min_count=args.min_count)
        plot_similarity_distribution("SBERT", "B", class_labels=["Accountancy Fee"], min_count=args.min_count)
        plot_similarity_distribution("SBERT", "B", class_labels=["Light, Heat & Power"], min_count=args.min_count)
 
 
if __name__ == "__main__":
    main()
 
