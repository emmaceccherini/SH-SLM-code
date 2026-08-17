
from utils import  load_and_filter
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def compute_class_similarities(types=["A", "B", "C"],
                               models=["SBERT", "mean_DEBERTA", "cls_DEBERTA"],
                               custom_labels=None,
                               min_count=1,
                               save_csv=True,
                               centering=False):
    """
    Compute within-class and between-class cosine similarities.
    Parameters
    ----------
    types : list of str
        Input types to iterate over.
    models : list of str
        Model names to iterate over.
    custom_labels : dict or None
        If provided, maps (input_type, model) -> array of labels to use
        instead of the native labels from load_and_filter.
        Can also be a single array, applied to all (type, model) combos.
    remove_less10 : bool
        Passed to load_and_filter.
    save_csv : bool
        Whether to save results to CSV files.
    centering : bool
        If True, subtract each cluster's centroid from its embeddings
        before computing cosine similarities (as in Cai et al. 2021, Section 3.4).
    Returns
    -------
    df_within, df_between : pd.DataFrame
    """
    within_results = {}
    between_results = {}
    for input_type in types:
        for model in models:
            embeddings, native_labels, source = load_and_filter(input_type, model, min_count=min_count)
            # Decide which labels to use
            if custom_labels is None:
                labels = native_labels
            elif isinstance(custom_labels, dict):
                labels = custom_labels.get((input_type, model), native_labels)
            else:
                labels = np.asarray(custom_labels)

            unique_labels = np.unique(labels)

            if centering:
                # Center each cluster by subtracting its centroid
                centered_embeddings = embeddings.copy()
                for label in unique_labels:
                    mask = labels == label
                    centroid = embeddings[mask].mean(axis=0)
                    centered_embeddings[mask] -= centroid
                sim_matrix = cosine_similarity(centered_embeddings)
            else:
                sim_matrix = cosine_similarity(embeddings)

            within_sims = {}
            between_sims = {}
            for label in unique_labels:
                mask = labels == label
                idx_in = np.where(mask)[0]
                idx_out = np.where(~mask)[0]
                sub = sim_matrix[np.ix_(idx_in, idx_in)]
                triu = np.triu_indices(len(idx_in), k=1)
                within_sims[label] = sub[triu].mean() if len(idx_in) > 1 else np.nan
                cross = sim_matrix[np.ix_(idx_in, idx_out)]
                between_sims[label] = cross.mean() if cross.size > 0 else np.nan
            within_results[(input_type, model)] = within_sims
            between_results[(input_type, model)] = between_sims
    # Build DataFrames
    within_rows = []
    between_rows = []
    for (input_type, model), sims in within_results.items():
        for label, val in sims.items():
            within_rows.append({"type": input_type, "model": model, "class": label, "within_cosine_sim": val})
    for (input_type, model), sims in between_results.items():
        for label, val in sims.items():
            between_rows.append({"type": input_type, "model": model, "origin_class": label, "between_cosine_sim": val})
    df_within = pd.DataFrame(within_rows)
    df_between = pd.DataFrame(between_rows)
    # print("=== Within-class ===")
    # print(df_within)
    # print("\n=== Between-class ===")
    # print(df_between)
    if save_csv:
        df_within.to_csv("within_class_cosine_similarities.csv", index=False)
        df_between.to_csv("between_class_cosine_similarities.csv", index=False)
    return df_within, df_between

def plot_similarity_distribution(model, input_type, class_labels=None, bins=50, kde=False, min_count=1):
    """
    Plot the distribution of within-class and between-class cosine similarities.

    Parameters
    ----------
    model : str
        One of "SBERT", "mean_DEBERTA", "cls_DEBERTA".
    input_type : str
        One of "A", "B", "C".
    class_labels : list or None
        List of class labels to include. If None, all classes are used.
    bins : int
        Number of histogram bins.
    kde : bool
        If True, overlay a KDE curve instead of a histogram.
    """
    embeddings, labels, source = load_and_filter(input_type, model, min_count=min_count)
    sim_matrix = cosine_similarity(embeddings)

    if class_labels is None:
        class_labels = list(np.unique(labels))
        class_desc = "all_classes"
    else:
        class_desc = class_labels[0]

    within_vals = []
    between_vals = []

    for label in class_labels:
        mask = labels == label
        idx_in = np.where(mask)[0]
        idx_out = np.where(~mask)[0]

        # Collect raw within-class pairwise similarities (upper triangle)
        if len(idx_in) > 1:
            sub = sim_matrix[np.ix_(idx_in, idx_in)]
            triu = np.triu_indices(len(idx_in), k=1)
            within_vals.extend(sub[triu].tolist())

        # Collect raw between-class similarities
        if idx_out.size > 0 and idx_in.size > 0:
            cross = sim_matrix[np.ix_(idx_in, idx_out)]
            between_vals.extend(cross.ravel().tolist())

    fig, ax = plt.subplots(figsize=(8, 5))

    label_desc = ", ".join(str(l) for l in class_labels) if len(class_labels) <= 5 else f"{len(class_labels)} classes"

    if kde:
        from scipy.stats import gaussian_kde
        if within_vals:
            density_w = gaussian_kde(within_vals)
            xs = np.linspace(min(within_vals + between_vals), max(within_vals + between_vals), 500)
            ax.plot(xs, density_w(xs), color="steelblue", label="Within-class")
            ax.fill_between(xs, density_w(xs), alpha=0.3, color="steelblue")
        if between_vals:
            density_b = gaussian_kde(between_vals)
            xs = np.linspace(min(within_vals + between_vals), max(within_vals + between_vals), 500)
            ax.plot(xs, density_b(xs), color="salmon", label="Between-class")
            ax.fill_between(xs, density_b(xs), alpha=0.3, color="salmon")
    else:
        if within_vals:
            ax.hist(within_vals, bins=bins, alpha=0.5, color="steelblue", label="Within-class", density=True)
        if between_vals:
            ax.hist(between_vals, bins=bins, alpha=0.5, color="salmon", label="Between-class", density=True)

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Within vs Between Cosine Similarity\nModel: {model} | Input: {input_type} | Classes: [{label_desc}]")
    ax.legend()
    plt.tight_layout()
    # save the plot
    # if class_labels is None:
    #     class_desc = "all_classes"
    # else:
    #     class_desc =class_labels[0] 
    plt.savefig(f"cosine_similarity_distribution_{model}_{input_type}_{class_desc}.png")
    print(f"Saved plot: cosine_similarity_distribution_{model}_{input_type}_{class_desc}.png")
    # plt.show()

def summarize_cosine_similarities(df_within, df_between, save_csv=True):
    """
    Compute average within-class and between-class cosine similarities per type/model.
    """
    avg_within = df_within.groupby(["type", "model"])["within_cosine_sim"].mean().reset_index(name="avg_within_cosine_sim")
    avg_between = df_between.groupby(["type", "model"])["between_cosine_sim"].mean().reset_index(name="avg_between_cosine_sim")
    avg_combined = avg_within.merge(avg_between, on=["type", "model"])
    if save_csv:
        avg_combined.to_csv("avg_within_between_cosine_similarities.csv", index=False)
    return avg_combined

