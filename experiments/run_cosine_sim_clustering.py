
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cosine_similarity import compute_class_similarities, summarize_cosine_similarities

import pickle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=40, help="Minimum label count for filtering (default: 1)")
    args = parser.parse_args()
    # load clustering results
    with open('clustering_results.pkl', 'rb') as f:
        clustering_results = pickle.load(f)

    custom_labels = {
        key: val['cluster_labels']
        for key, val in clustering_results.items()
    }

    df_within, df_between = compute_class_similarities(centering=False, custom_labels=custom_labels, min_count=args.min_count, save_csv=False)
    avg_combined = summarize_cosine_similarities(df_within, df_between, save_csv=False)
    print(avg_combined)


    # repeat with centering
    df_within_centered, df_between_centered = compute_class_similarities(centering=True, custom_labels=custom_labels, min_count=args.min_count, save_csv=False)
    avg_combined_centered = summarize_cosine_similarities(df_within_centered, df_between_centered, save_csv=False)
    print(avg_combined_centered)   

if __name__ == "__main__":
    main()