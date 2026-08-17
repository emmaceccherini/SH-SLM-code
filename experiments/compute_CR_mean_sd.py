#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from classification import *
import torch
#%%


seeds = [42, 123, 24, 23, 52]
dfs = []
for s in seeds:
    res = torch.load(f".../sbert_ft_B_6_{s}.pkl", weights_only=False)

    logits = np.asarray(res["logits"])
    labels = np.asarray(res["labels"])
    preds = logits.argmax(axis=-1)

    id2label = res["id2label"]
    # class ids in sorted order, mapped to their names
    class_ids = sorted(id2label.keys())
    target_names = [id2label[i] for i in class_ids]

    report = classification_report(
        labels,
        preds,
        labels=class_ids, output_dict=True, target_names=target_names
    )
    report_df = pd.DataFrame(report).transpose()
    dfs.append(report_df)

#%%
# Preserve original row order rather than the alphabetical order groupby produces.
row_order = dfs[0].index.tolist()

# Stack all seeds: MultiIndex (seed, label) x metrics (precision, recall, f1-score, support).
combined = pd.concat(dfs, keys=seeds, names=["seed", "label"])

grouped = combined.groupby(level="label")

mean_df = grouped.mean().reindex(row_order)
var_df  = grouped.var(ddof=1).reindex(row_order)   
std_df  = grouped.std(ddof=1).reindex(row_order)

# Formatted "mean ± std" table for reporting.
formatted = mean_df.copy()
for col in mean_df.columns:
    if col == "support":
        formatted[col] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(mean_df[col], std_df[col])]
    else:
        formatted[col] = [f"{m:.2f} ± {s:.2f}" for m, s in zip(mean_df[col], std_df[col])]





