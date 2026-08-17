#%%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classification import *
import glob
import os
import torch
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import re
#%%

def parse_inj(name):
    m = re.search(r"_inj([\d.]+)_seed", name)
    return float(m.group(1))

def load_files(directory, name):
    """
    Loads all files in `directory` starting with 'sbert' and ending with 'C1.pkl'.
    Returns a dict mapping filename -> loaded object.
    """
    pattern = os.path.join(directory, name)
    results = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        print(f"Loading {name}...")
        results[name] = torch.load(path, weights_only=False)
    return results
#%%

test_clients = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

models = ["sbert", "deberta"]
logits = {}   
labels = {}

for model in models:
    for client in test_clients:
        res = load_files("gen_results", f"{model}_*test-{client}*.pkl")
        for name, data in res.items():
            key = (model, parse_inj(name))
            logits.setdefault(key, []).append(np.asarray(data["logits"]))
            labels.setdefault(key, []).append(np.asarray(data["labels"]))
        del res


logits = {k: np.concatenate(v) for k, v in logits.items()}
labels = {k: np.concatenate(v) for k, v in labels.items()}


def compute_metrics(logits, labels):
    """
    logits: array-like of shape (N, num_classes) — raw model outputs
    labels: array-like of shape (N,) — integer class labels
    Returns dict with accuracy, macro F1, and weighted F1.
    """
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    preds = logits.argmax(axis=-1)
    

    return {
        "accuracy": (preds == labels).mean(),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }
# %%

metrics_df = pd.DataFrame(columns=["model", "inj", "accuracy", "macro_f1", "weighted_f1"])
for (model, inj) in logits.keys():
    metrics = compute_metrics(logits[(model, inj)], labels[(model, inj)])
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([{
            "model": model,
            "inj": inj,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }])
    ], ignore_index=True)

metrics_df = metrics_df.sort_values(["model", "inj"]).reset_index(drop=True)# %%
#%%
# load the baseline results from the csv files
baseline_results = {}
for e in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    df = pd.read_csv(f"baseline_inject_{e*100:.0f}.csv")
    baseline_results[e] = {
        "accuracy": df["accuracy"].mean(),
        "macro_f1": df["macro_f1"].mean(),
        "weighted_f1": df["weighted_f1"].mean(),
    }
#%%
colors = [
    "#FB8072",
    "#8DD3C7"
]# create a plot
vl1 = 0.939
vl2 =0.916
vl3 = 0.87
vl4 = 0.39
sbert = metrics_df[metrics_df["model"] == "sbert"]
deberta = metrics_df[metrics_df["model"] == "deberta"]

plt.plot(sbert["inj"], sbert["macro_f1"], marker='o', linestyle='-', color=colors[0],
         label='SBERT - new client', linewidth=4, markersize=8)
plt.plot(deberta["inj"], deberta["macro_f1"], marker='o', linestyle='-', color=colors[1],
         label='DeBERTa - new client', linewidth=4, markersize=8)
plt.plot(list(baseline_results.keys()), [baseline_results[e]["macro_f1"] for e in baseline_results.keys()], marker='o', linestyle='-', color="gray",
         label='vendor baseline', linewidth=4, markersize=8)

plt.xticks(list(sbert["inj"]), fontsize=15)

plt.axhline(y=vl1, color=colors[0], linestyle='--', label='SBERT FT baseline', linewidth=4, alpha=0.7)
plt.axhline(y=vl2, color=colors[1], linestyle='--', label='DeBERTa FT baseline', linewidth=4, alpha=0.7)
# plt.axhline(y=vl3, color='gray', linestyle='--', label='Vendor baseline', linewidth=4, alpha=0.7)
plt.axhline(y=vl4, color='darkgray', linestyle='--', label='LLM baseline', linewidth=4, alpha=0.7)

plt.xlabel("Injection proportion", fontsize=17)
plt.ylabel("Macro F1", fontsize=17)
plt.yticks(fontsize=15)
# plt.title("Accuracy over multiple runs")
# plt.legend()
# make the legend outside the plot in the center below the plot
# change the fontsize to 15 for 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=17)
plt.savefig("gen_experiment_baseline.png", dpi=150, bbox_inches="tight")
# %%