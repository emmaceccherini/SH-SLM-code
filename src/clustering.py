import pandas as pd
import pickle 
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from utils import load_and_filter, get_keep_indices
import matplotlib.pyplot as plt

from config import consolidated_csv, CLIENTS, input_csv


def run_clustering(input_types=None, target_models=None, max_clusters= 15, save= False, plots = False, min_count=1):
    """
    Run clustering for specified types and models.
    Defaults to all types and models if not specified.
    """
    if input_types is None:
        input_types = ["A", "B", "C"]
    if target_models is None:
        target_models = ["SBERT", "mean_DEBERTA", "cls_DEBERTA"]

    # Ensure inputs are lists
    if isinstance(input_types, str):
        input_types = [input_types]
    if isinstance(target_models, str):
        target_models = [target_models]
    cache = {}
    clustering_results = {}
    for input_type in input_types:
        for model in target_models:
            print(f'Clustering for {input_type} with {model}')
            embeddings, labels, _ = load_and_filter(input_type, model, min_count=min_count)
            silhouette_scores = []
            for n_clusters in range(2, max_clusters+1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append((n_clusters, silhouette_avg))
            
            cache[(input_type, model)] = silhouette_scores
            best_n_clusters, best_score = max(silhouette_scores, key=lambda x: x[1])
            # print(f'Best number of clusters: {best_n_clusters} with silhouette score: {best_score:.4f}')
            # do the best clustering
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            clustering_results[(input_type, model)] = {
                'best_n_clusters': best_n_clusters,
                'best_score': best_score,
                'cluster_labels': cluster_labels }
            
            # ari_score = adjusted_rand_score(labels, cluster_labels)
            # print(f'{input_type} - {model}: Adjusted Rand Index = {ari_score:.2f}')

    if save: 
        with open('clustering_results.pkl', 'wb') as f:
            pickle.dump(clustering_results, f)

    for (input_type, model), result in clustering_results.items():
        print(f'{input_type} - {model}: Best n_clusters = {result["best_n_clusters"]}, Silhouette Score = {result["best_score"]:.4f}')

    if plots:
        for (input_type, model), result in clustering_results.items():
            silhouette_scores = cache[(input_type, model)]

            n_clusters, scores = zip(*silhouette_scores)
            plt.figure(figsize=(10, 6))
            plt.plot(n_clusters, scores, marker='o')
            plt.title(f'Silhouette Scores for {input_type} with {model}')
            plt.xlabel('Number of Clusters')
            plt.ylabel('Silhouette Score')
            plt.xticks(range(min(n_clusters), max(n_clusters) + 1, 10))
            plt.grid()
            plt.savefig(f'silhouette_{input_type}_{model}.png')
            # plt.show()
    return clustering_results

def evaluate_clustering(clustering_results, min_count=1):
    """
    Evaluate clustering results using Adjusted Rand Index and Normalized Mutual Information.
    """
    for (input_type, model), result in clustering_results.items():
        cluster_labels = result['cluster_labels']
        _, labels, _ = load_and_filter(input_type, model, min_count=min_count)
        ari_score = adjusted_rand_score(labels, cluster_labels)
        print(f'{input_type} - {model}: Adjusted Rand Index = {ari_score:.2f}')
        nmi_score = normalized_mutual_info_score(labels, cluster_labels)
        print(f'{input_type} - {model}: Normalized Mutual Information = {nmi_score:.2f}')

def compare_clustering_vendors(clustering_results, min_count=1):
    # Transaction ID -> Vendor Name lookup, per client
    vendor_lookup = {}
    for c in CLIENTS:
        cons = pd.read_csv(consolidated_csv(c))
        cons["Vendor Name"] = cons["Vendor Name"].fillna("Unknown")
        vendor_lookup[c] = dict(zip(cons["Transaction ID"], cons["Vendor Name"]))

    for (input_type, model), result in clustering_results.items():
        cluster_labels = result["cluster_labels"]
        keep_indices   = get_keep_indices(min_count=min_count, input_type=input_type)

        vendors = []
        for i, c in enumerate(CLIENTS):
            inp  = pd.read_csv(input_csv(c, input_type))
            kept = sorted(keep_indices[c])
            txn_ids = inp.iloc[kept]["Transaction ID"].tolist()
            vendors.extend(vendor_lookup[c].get(t, "Unknown") for t in txn_ids)

        assert len(vendors) == len(cluster_labels), (
            f"{input_type}/{model}: {len(vendors)} vendors vs "
            f"{len(cluster_labels)} cluster labels"
        )

        ari = adjusted_rand_score(vendors, cluster_labels)
        nmi = normalized_mutual_info_score(vendors, cluster_labels)
        print(f"{input_type} - {model}: ARI (vs vendor) = {ari:.2f}, NMI (vs vendor) = {nmi:.2f}")

def vendor_vs_label_agreement(input_type="B", min_count=1):
    """How well does Vendor Name align with the true Label, on its own?"""
    vendors_all, labels_all = [], []
    keep_indices = get_keep_indices(min_count=min_count, input_type=input_type)
    for c in CLIENTS:
        cons = pd.read_csv(consolidated_csv(c))
        cons["Vendor Name"] = cons["Vendor Name"].fillna("Unknown")
        vendor_lookup = dict(zip(cons["Transaction ID"], cons["Vendor Name"]))

        inp  = pd.read_csv(input_csv(c, input_type))
        kept = sorted(keep_indices[c])
        sub  = inp.iloc[kept]
        vendors_all.extend(vendor_lookup.get(t, "Unknown") for t in sub["Transaction ID"])
        labels_all.extend(sub["Label"].tolist())

    print(f"{input_type}: ARI(vendor, label) = {adjusted_rand_score(vendors_all, labels_all):.2f}, "
          f"NMI(vendor, label) = {normalized_mutual_info_score(vendors_all, labels_all):.2f}")