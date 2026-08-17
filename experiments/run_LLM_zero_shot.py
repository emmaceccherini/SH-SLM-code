#%%
import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CLIENTS
from classification import *
import re 

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
#%%
MODEL_ID = "Qwen/Qwen3-4B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)
model.eval()

#%%
categories_desc = [
    "Accountancy Fee: Fees for preparation of statutory and/or management accounts, bookkeeping, tax compliance and related non-audit advisory services. Excludes statutory audit. ",
    "Cleaning of Premises: Janitorial costs of keeping premises clean — cleaning staff or contractors, services and cleaning consumables. Excludes repair and upkeep of premises, which is maintenance (see Repairs and Renewals).",
    "Directors' Loan Account: Related-party control account for amounts a director personally lends to, or withdraws from, the company other than salary, dividend or reimbursed expenses. May be in credit (company owes the director — a liability) or overdrawn (director owes the company — an asset); overdrawn balances attract a s455 CTA 2010 charge and beneficial-loan implications.",
    "Light, Heat & Power: Utility costs — electricity, gas, heating and power — consumed in operating business premises",
    "Motor Repairs and Servicing: Maintenance, servicing and repair of company vehicles.",
    "Petrol and Oil:Fuel and lubricant costs incurred in operating business motor vehicles.",
    "Purchases : Cost of goods, raw materials or components bought for resale or for consumption in production during the period. A profit-and-loss item forming part of cost of sales; it records the buying flow, not stock held. Excludes capital assets and overhead services.",
    "Stationery & Postage: Office consumables, printing materials and postal or courier charges.",
    "Sub: Abbreviation for Subcontractor. Payments for subcontractor or subcontract labour directly attributable to revenue-generating work.",
    "Subscriptions : Recurring professional-body memberships, journals and software or service subscriptions of an administrative nature.",
    "Telephone, Fax & Internet: Communication costs: telephone, broadband, internet access and related charges.",
    "Transport, Freight & Carriage: Inwards - Cost of transporting purchased goods or materials into the business. Added to the cost of purchases and therefore part of cost of sales. Outwards - Cost of delivering finished goods out to customers. A distribution/selling overhead, not cost of sales.",
]


CATEGORIES = [
    "Accountancy Fee",
    "Cleaning of Premises",
    "Directors' Loan Account",
    "Light, Heat & Power",
    "Motor Repairs and Servicing",
    "Petrol and Oil",
    "Purchases",
    "Stationery & Postage",
    "Sub",
    "Subscriptions",
    "Telephone, Fax & Internet",
    "Transport, Freight & Carriage",
]

#%%

UNPARSED = "__unparsed__"


def _norm(s: str) -> str:
    """Lowercase, collapse any run of non-alphanumerics to a single space, trim."""
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


#  Map normalized label -> canonical label.
_NORM_CATS = {_norm(c): c for c in CATEGORIES}
# Longest normalized label first 
_CATS_BY_LEN = sorted(CATEGORIES, key=lambda c: len(_norm(c)), reverse=True)


def parse_label(text: str) -> str:
    norm = _norm(text)
    if not norm:
        return UNPARSED

    
    if norm in _NORM_CATS:
        return _NORM_CATS[norm]


    anchored = re.sub(r"^category\s+", "", norm)
    for cat in _CATS_BY_LEN:
        nc = _norm(cat)
        if re.match(rf"{re.escape(nc)}\b", anchored):
            return cat

    found = {
        cat for cat in _CATS_BY_LEN
        if re.search(rf"\b{re.escape(_norm(cat))}\b", norm)
    }
    # Drop "Sub" if "Subscriptions" is also present (substring family).
    if "Subscriptions" in found:
        found.discard("Sub")
    if len(found) == 1:
        return next(iter(found))

    return UNPARSED
#%%
accuracy_scores = []
macro_f1_scores = []
weighted_f1_scores = []
seeds = [42, 123, 24, 23, 52]
for s in seeds:
    outputs_text = []
    data = create_dataset_dict(input_type = "B", clients = CLIENTS, return_ids = True, seed = s, )
    train_dataset = pd.concat([data["train"], data["val"]], ignore_index=True)
    test_dataset = data["test"]     
    gold = test_dataset["Label"].tolist()
    unknown = set(gold) - set(CATEGORIES)
    assert not unknown, f"Gold labels not in CATEGORIES: {unknown}"


    for i in range(len(test_dataset)):
        prompt = f"You are an expert UK bookkeeper assigning invoices to General Ledger categories. Choose exactly one category from the list. Definitions: {', '.join(categories_desc)}. invoice: {test_dataset.iloc[i]['Text Input']}. Only answer with the name of the category, do not answer with any other text."

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)


        output = model.generate(
            **input_ids,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=100,
            do_sample=False
        )

        gen = output[0][input_ids["input_ids"].shape[-1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        
        outputs_text.append(parse_label(text))

    gold_parsed = [parse_label(g) for g in gold]   

    n_unparsed = outputs_text.count("__unparsed__")
    print(f"Unparsed predictions: {n_unparsed}/{len(outputs_text)}")

    acc = accuracy_score(gold_parsed, outputs_text)
    macro_f1 = f1_score(gold_parsed, outputs_text, labels=CATEGORIES,
                        average="macro", zero_division=0)
    weighted_f1 = f1_score(gold_parsed, outputs_text, labels=CATEGORIES,
                            average="weighted", zero_division=0)
    print(classification_report(gold_parsed, outputs_text, labels=CATEGORIES,
                                    zero_division=0))
    # save the classification report to a file
    report = classification_report(gold_parsed, outputs_text, labels=CATEGORIES,
                                    zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"CL_LLM_zero_shot_seed_{s}.csv", index=True)
    print(f"Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

    accuracy_scores.append(acc)
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)


# %%
print(f"Accuracy:    {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Macro F1:    {np.mean(macro_f1_scores):.4f} ± {np.std(macro_f1_scores):.4f}")
print(f"Weighted F1: {np.mean(weighted_f1_scores):.4f} ± {np.std(weighted_f1_scores):.4f}")

# %%
#%%
import numpy as np
import pandas as pd

seeds = [42, 123, 24, 23, 52]

# Load each report; first column is the row label (category / avg rows).
dfs = [pd.read_csv(f"CL_LLM_zero_shot_seed_{s}.csv", index_col=0) for s in seeds]

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

mean_df.to_csv("CL_LLM_zero_shot_mean.csv")
var_df.to_csv("CL_LLM_zero_shot_variance.csv")
formatted.to_csv("CL_LLM_zero_shot_mean_std.csv")
# %%
