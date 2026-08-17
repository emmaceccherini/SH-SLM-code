import argparse

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from clustering import run_clustering, evaluate_clustering, compare_clustering_vendors, vendor_vs_label_agreement


 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_clusters", type=int, default=50, help="Maximum number of clusters to evaluate")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    parser.add_argument("--min-count", type=int, default=1, help="Minimum label count for filtering (default: 1)")

    args = parser.parse_args()
    
    clustering_results = run_clustering(max_clusters=args.max_clusters, save=True, plots=True, min_count=args.min_count)
    evaluate_clustering(clustering_results, min_count=args.min_count)
    compare_clustering_vendors(clustering_results, min_count=args.min_count)
    vendor_vs_label_agreement(min_count=args.min_count)

if __name__ == "__main__":
    main()