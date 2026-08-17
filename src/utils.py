
import pickle
import pandas as pd
from collections import Counter
import numpy as np
import sklearn

from config import (
    CLIENTS,
    consolidated_csv,
    embeddings_pkl,
    input_csv,
    merged_csv,
)

def get_keep_indices(min_count=1, input_type="B"):

    labels_raw = {
        client: pd.read_csv(input_csv(client, input_type))["Label"].tolist()
        for client in CLIENTS
    }
    all_labels = [l for cl_labels in labels_raw.values() for l in cl_labels]
    label_counts = Counter(all_labels)
    labels_to_keep = {label for label, count in label_counts.items() if count >= min_count}
    return {
        name: {i for i, label in enumerate(labels_raw[name]) if label in labels_to_keep}
        for name in CLIENTS
    }

def load_data(clients: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged_dfs, consolidated_dfs = [], []
    for c in clients:
        merged_dfs.append(pd.read_csv(merged_csv(c)))
        consolidated_dfs.append(pd.read_csv(consolidated_csv(c)))
    return pd.concat(merged_dfs, ignore_index=True), pd.concat(consolidated_dfs, ignore_index=True)


def load_and_filter(input_type, model, min_count=1, vendor_name=False):
    labels_raw = {}
    embeddings = {}
    vendors_raw = {}
    for client in CLIENTS:
        with open(embeddings_pkl(client, input_type, model), "rb") as f:
            embeddings[client] = pickle.load(f)

        df = pd.read_csv(input_csv(client, input_type))
        labels_raw[client] = df["Label"].tolist()

        if vendor_name:
            vendors_raw[client] = (
                df.iloc[:, 1].astype(str).str.split(",", n=1).str[0].str.strip().tolist()
            )

        assert len(embeddings[client]) == len(labels_raw[client]), (
            f"{client} {input_type} {model}: "
            f"{len(embeddings[client])} embeddings vs {len(labels_raw[client])} labels"
        )

    if min_count is not None:
        keep_indices = get_keep_indices(min_count=min_count, input_type=input_type)
        for name in CLIENTS:
            keep = keep_indices[name]
            embeddings[name] = [e for i, e in enumerate(embeddings[name]) if i in keep]
            labels_raw[name] = [l for i, l in enumerate(labels_raw[name]) if i in keep]
            if vendor_name:
                vendors_raw[name] = [v for i, v in enumerate(vendors_raw[name]) if i in keep]

    all_embeddings = np.concatenate([embeddings[name] for name in CLIENTS], axis=0)
    all_labels     = np.concatenate([labels_raw[name] for name in CLIENTS], axis=0)
    source         = np.concatenate([[i]*len(embeddings[c]) for i, c in enumerate(CLIENTS)], axis=0)

    if vendor_name:
        all_vendor_names = np.concatenate([vendors_raw[name] for name in CLIENTS], axis=0)
        return all_embeddings, all_labels, source, all_vendor_names
    return all_embeddings, all_labels, source