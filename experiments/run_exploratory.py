#!/usr/bin/env python3
"""Analyze merged/consolidated client data with summary stats and plots."""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import CLIENTS, consolidated_csv, merged_csv
from utils import load_data

def plot_category_counts(consolidated_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    order = consolidated_df["Label"].value_counts().index
    sns.countplot(x="Label", data=consolidated_df, order=order)
    plt.title("Number of data points in each category")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    for p in plt.gca().patches:
        plt.gca().annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center", va="center", fontsize=10,
            xytext=(0, 5), textcoords="offset points",
        )
    plt.tight_layout()
    # save the plot as an image file
    plt.savefig("category_distribution.png")
    # plt.show()


def print_summary(merged_df: pd.DataFrame, consolidated_df: pd.DataFrame) -> None:
    print(f"Number of data points: {len(consolidated_df)}")
    print(f"Number of categories: {consolidated_df['Label'].nunique()}")

    purchase_pct = (consolidated_df["Label"] == "Purchases").mean() * 100
    print(f"Percentage of purchases: {purchase_pct:.2f}%")

    mean_li = len(merged_df) / merged_df["Transaction ID"].nunique()
    print(f"Average number of line items: {mean_li:.2f}")

    print("\nMissing values per column:")
    for col in consolidated_df.columns:
        pct = consolidated_df[col].isnull().mean() * 100
        print(f"  {col}: {pct:.2f}%")


def tokenizer_stats(consolidated_df: pd.DataFrame) -> None:
    from transformers import AutoTokenizer
    checkpoint= "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    line_items = (
        consolidated_df["Line Items"]
        # .str.replace(r"Line Quantity:", "", regex=True)
        # .str.replace(r"Line Amount:", "", regex=True)
        # .str.replace(r"\d+:", "", regex=True)
        .tolist()
    )
    tokenized = tokenizer(line_items)
    mean_len = np.mean([len(ids) for ids in tokenized["input_ids"]])
    print(f"Average tokens in line items: {mean_len:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", nargs="+", default=CLIENTS,
                        help="List of clients to analyze (default: all)")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip tokenizer stats (avoids transformers dependency)")
    parser.add_argument("--no-plot", action="store_true", help="Skip the bar plot")
    args = parser.parse_args()

    merged_df, consolidated_df = load_data(args.clients)
    print_summary(merged_df, consolidated_df)

    if not args.skip_tokenizer:
        tokenizer_stats(consolidated_df)
    if not args.no_plot:
        plot_category_counts(consolidated_df)

if __name__ == "__main__":
    main()