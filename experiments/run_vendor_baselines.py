#%%
from experiments.run_generalisability import create_dataset_dict_gen

import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CLIENTS
from classification import *
import re 

#%%
accuracy_scores = []
macro_f1_scores = []
weighted_f1_scores = []
seeds = [42, 123, 24, 23, 52]
for s in seeds:
    data = create_dataset_dict(input_type = "A", clients = CLIENTS, return_ids = True, seed = s, )
    train_dataset = pd.concat([data["train"], data["val"]], ignore_index=True)
    test_dataset = data["test"]

    # get the text before the first comma in the "Vendor" column
    train_dataset["vendor"] = train_dataset["Text Input"].astype(str).str.split(",", n=1).str[0].str.strip()
    test_dataset["vendor"] = test_dataset["Text Input"].astype(str).str.split(",", n=1).str[0].str.strip()


    # is there a vendor in the test set that is not in the train set?
    test_vendors = set(test_dataset["vendor"].unique())
    train_vendors = set(train_dataset["vendor"].unique())
    #print(test_vendors - train_vendors)


    vendor_to_label = (
        train_dataset.groupby(['vendor', 'Label'])
            .size()
            .reset_index(name='n')
            .sort_values(['vendor', 'n'], ascending=[True, False])
            .drop_duplicates('vendor')
            .set_index('vendor')['Label']
            .to_dict()
    )

    # for vendors not seen in training  global majority label
    fallback_label = train_dataset['Label'].mode()[0]


    y_true = test_dataset['Label']
    y_pred = test_dataset['vendor'].map(vendor_to_label).fillna(fallback_label)

    # Metrics
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # print(f"Accuracy:    {acc:.4f}")
    # print(f"Macro F1:    {macro_f1:.4f}")
    # print(f"Weighted F1: {weighted_f1:.4f}")

    accuracy_scores.append(acc)
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)

    report = classification_report(y_true, y_pred,
                                    zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"CL_vendor_bl_seed_{s}.csv", index=True)
#%%
import numpy as np
import pandas as pd

seeds = [42, 123, 24, 23, 52]

# Load each report; first column is the row label (category / avg rows).
dfs = [pd.read_csv(f"CL_vendor_bl_seed_{s}.csv", index_col=0) for s in seeds]

# Preserve original row order rather than the alphabetical order groupby produces.
row_order = dfs[0].index.tolist()

# Stack all seeds: MultiIndex (seed, label) x metrics (precision, recall, f1-score, support).
combined = pd.concat(dfs, keys=seeds, names=["seed", "label"])

grouped = combined.groupby(level="label")
#%%
mean_df = grouped.mean().reindex(row_order)
var_df  = grouped.var(ddof=1).reindex(row_order)   # sample variance (n-1); use ddof=0 for population
std_df  = grouped.std(ddof=1).reindex(row_order)

# Formatted "mean ± std" table for reporting.
formatted = mean_df.copy()
for col in mean_df.columns:
    if col == "support":
        formatted[col] = [f"{m:.1f} ± {s:.1f}" for m, s in zip(mean_df[col], std_df[col])]
    else:
        formatted[col] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_df[col], std_df[col])]

print("Mean:\n", mean_df.round(4), "\n")
print("Variance:\n", var_df.round(6), "\n")
print("Mean ± Std:\n", formatted)

# mean_df.to_csv("CL_vendor_bl_mean.csv")
# var_df.to_csv("CL_vendor_bl_variance.csv")
# formatted.to_csv("CL_vendor_bl_mean_std.csv")


#%%
# print the mean and std of the metrics across the seeds
print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Macro F1:    {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}")
print(f"Weighted F1: {np.mean(weighted_f1_scores):.4f} ± {np.std(weighted_f1_scores):.4f}")
#%%
print(classification_report(y_true, y_pred))
# %%
# leave one out 

fracs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
for e in fracs: 
    accuracy_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []
    print(f"Injecting {e*100:.0f}% of data.")
    for c in CLIENTS:
        clients_test = [c]
        clients_train = [x for x in CLIENTS if x != c]
        data = create_dataset_dict_gen("A", inject_frac=e, clients_train = clients_train, clients_test = clients_test)

        # transform them to pandas dataframes
        for k, v in data.items():
            data[k] = pd.DataFrame(v)

        train_dataset = pd.concat([data["train"], data["val"]], ignore_index=True)
        test_dataset = data["test"]
        # get the text before the first comma in the "Vendor" column
        train_dataset["vendor"] = train_dataset["text"].astype(str).str.split(",", n=1).str[0].str.strip()
        test_dataset["vendor"] = test_dataset["text"].astype(str).str.split(",", n=1).str[0].str.strip()

        #print(len(np.unique(train_dataset["vendor"], return_counts=False)))


        # is there a vendor in the test set that is not in the train set?
        test_vendors = set(test_dataset["vendor"].unique())
        train_vendors = set(train_dataset["vendor"].unique())
        # train_dataset["vendor"] = train_dataset["vendor"].map(normalize_vendor)
        # test_dataset["vendor"]  = test_dataset["vendor"].map(normalize_vendor)

        #print(test_vendors - train_vendors)
        vendor_to_label = (
            train_dataset.groupby(['vendor', 'label'])
                .size()
                .reset_index(name='n')
                .sort_values(['vendor', 'n'], ascending=[True, False])
                .drop_duplicates('vendor')
                .set_index('vendor')['label']
                .to_dict()
        )

        # for vendors not seen in training  global majority label
        fallback_label = train_dataset['label'].mode()[0]


        y_true = test_dataset['label']
        y_pred = test_dataset['vendor'].map(vendor_to_label).fillna(fallback_label)

        # Metrics
        acc         = accuracy_score(y_true, y_pred)
        macro_f1    = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        # print(f"Accuracy:    {acc:.4f}")
        # print(f"Macro F1:    {macro_f1:.4f}")
        # print(f"Weighted F1: {weighted_f1:.4f}")

        accuracy_scores.append(acc)
        macro_f1_scores.append(macro_f1)
        weighted_f1_scores.append(weighted_f1)


    print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
    print(f"Macro F1:    {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}")
    print(f"Weighted F1: {np.mean(weighted_f1_scores):.4f} ± {np.std(weighted_f1_scores):.4f}")
    #save the results to a csv file
    results_df = pd.DataFrame({
        "accuracy": accuracy_scores,
        "macro_f1": macro_f1_scores,
        "weighted_f1": weighted_f1_scores,
    })
    results_df.to_csv(f"baseline_inject_{e*100:.0f}.csv", index=False)
# %%
