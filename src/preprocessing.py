#!/usr/bin/env python3
"""
Data pre processing. 

In a directory save CSV files dowloaded from 
... from the "completed" tab of each client. 
You should have a file for each page of the completed tab, 
call these files ClientName_categories_p1.csv, ClientName_categories_p2.csv, etc.

In the same directory you should save the line items files, 
as obtained by the data scraper. These should be named 
ClientName_lineitems_1.csv, ClientName_line_items_2.csv, etc.

This script will return two files for each client. 
1) ClientName_merged.csv: This file has a line for each unique 
line item (*not invoice*), the invoice information pertaining to one or more line
item is repeated for each line item. This fie has the follwoing columns:
Transaction ID, Invoice Date, Tax, Currency, Line Item, Line Item Description, 
Line Quantity, Line Amount, Vendor Name, Label, Account Code, Price. 

2) ClientName_consolidated.csv: This file has a line for each unique invoice,
the line items pertaining to the same invoice are consolidated in a single line.
This file has the following columns: 'Transaction ID', 'Invoice Date', 'Tax', 'Currency', 'Line Items',
'Vendor Name', 'Label', 'Account Code', 'Price'. Where the 'Line Items' column 
contains a string with the line items of the invoice, in the format: "
Line Item Number: Line Item Description Line Quantity: Line Quantity Line Amount: Line Amount ; ...". 


"""

import argparse
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd


def find_files(directory: Path, prefix: str, suffix: str) -> list[Path]:
    """
    Find files in directory matching pattern: {prefix}{suffix}*.csv
    
    Matches patterns like:
        C1_categories.csv, C1_categories1.csv, C1_categories_1.csv, etc.
    """
    pattern = f"{prefix}{suffix}*.csv"
    files = sorted(directory.glob(pattern))
    return files


def clean_cat_df(df):
    """Clean and transform the transaction dataframe."""
    df = df.copy()
    
    # Remove rows where Account Name is null (before renaming)
    df = df.dropna(subset=['Account Name'])
    
    # Rename columns
    df = df.rename(columns={
        'Contact Name': 'Vendor Name',
        'Account Name': 'Label'
    })

    
    # Standardize Invoice Date format, then fill nulls
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce', dayfirst=True)
    df['Invoice Date'] = df['Invoice Date'].dt.strftime('%d-%m-%Y') 
    
    # Drop unwanted columns
    cols_to_drop = [
        'Transaction Type Name', 'File Name', 'Invoice No', 
        'Due Date', 'Vat Included', 'Currency Rate', 
        'Tax Name', 'Vat Rate'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # If there are NaN values in the Account Code column raise an error
    if df['Account Code'].isnull().any():
        raise ValueError("Error: 'Account Code' column contains null values. Please check the data.")
    else:
        #convert the Account Code column to integer
        df['Account Code'] = df['Account Code'].astype(int)
    
    return df

def clean_lineitems_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform line items DataFrame by removing, renaming, and merging columns.
    
    Parameters:
        df: Input DataFrame with line items data
        
    Returns:
        Transformed DataFrame
    """
    # Work on a copy to avoid modifying the original
    df = df.copy()
    
    # Columns to remove
    cols_to_drop = [
        'Type', 'File Name', 'Invoice Date', 'Expense Category',
        'Gross Amount (Header)', 'Net Amount (Header)', 'Tax Amount (Header)',
        'Vendor Name', 'Line Tax Amount'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Rename columns
    rename_map = {
        'Supplier': 'Invoice Date',
        'Total': 'Supplier',
        'Tax': 'Total',
        'Category': 'Tax',
        'Line Number': 'Line Item',
        'Line Description': 'Line Item Description'
    }
    df = df.rename(columns=rename_map)
    
    # Merge Line Net Amount and Line Gross Amount into Line Amount
    def merge_line_amounts(row, idx):
        net = row.get('Line Net Amount')
        gross = row.get('Line Gross Amount')
        
        # Check if values are empty (None, NaN, or empty string)
        net_empty = pd.isna(net) or net == ''
        gross_empty = pd.isna(gross) or gross == ''
        
        if net_empty and gross_empty:
            return np.float64('nan')   
        elif net_empty:
            return gross
        elif gross_empty:
            return net
        else:
            # Both have values - compare them
            if net == gross:
                return net
            else:
                print(f"Warning: Row {idx}, Transaction ID '{row.get('Transaction ID')}' - "
                      f"Line Net Amount ({net}) differs from Line Gross Amount ({gross}). "
                      f"Using Gross Amount.")
                return gross
    
    df['Line Amount'] = [merge_line_amounts(row, idx) for idx, row in df.iterrows()]
    
    # Drop the original columns
    df = df.drop(columns=['Line Net Amount', 'Line Gross Amount'], errors='ignore')
    
    return df


def load_and_concat(files: list[Path]) -> pd.DataFrame:
    """Load multiple CSV files and concatenate into single DataFrame."""
    if not files:
        raise FileNotFoundError("No matching files found.")
    
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def consolidate_transactions(df):
    """
    Consolidate rows with the same Transaction ID into single rows.
    
    - Combines line item information into a single 'Line Items' column
    - Warns if header fields (Invoice Date, Tax, Currency) differ across rows
    - Warns if footer fields (Vendor Name, Tax Amount, Label, Account Code, Price) differ
    - Uses first row values when conflicts exist
    """
    
    # Columns to check for consistency (should be same across all rows of a transaction)
    header_cols = ['Invoice Date', 'Tax', 'Currency']
    footer_cols = ['Vendor Name', 'Label', 'Account Code', 'Price']
    
    consolidated_rows = []
    
    # Group by Transaction ID
    for txn_id, group in df.groupby('Transaction ID', sort=False):
        first_row = group.iloc[0]
        
        # Check header columns for consistency
        for col in header_cols:
            unique_vals = group[col].dropna().unique()
            if len(unique_vals) > 1:
                warnings.warn(
                    f"Transaction {txn_id}: '{col}' has inconsistent values: {unique_vals.tolist()}. "
                    f"Using first row value: {first_row[col]}"
                )
        
        # Check footer columns for consistency
        for col in footer_cols:
            unique_vals = group[col].dropna().unique()
            if len(unique_vals) > 1:
                warnings.warn(
                    f"Transaction {txn_id}: '{col}' has inconsistent values: {unique_vals.tolist()}. "
                    f"Using first row value: {first_row[col]}"
                )
        
        # Build the combined Line Items string
        line_items_parts = []
        for _, row in group.iterrows():
            # Get line item number as integer if possible
            line_num = row['Line Item']
            if pd.notna(line_num):
                line_num = int(line_num) if float(line_num).is_integer() else line_num
            else:
                line_num = len(line_items_parts) + 1  # fallback numbering
            
            # Check what content we have
            desc = row['Line Item Description']
            qty = row['Line Quantity']
            amt = row['Line Amount']
            
            has_desc = pd.notna(desc) and desc != ''
            has_qty = pd.notna(qty) and qty != ''
            has_amt = pd.notna(amt) and amt != ''
            
            # Only create item string if there's actual content
            if has_desc or has_qty or has_amt:
                if has_desc:
                    item_str = f"{desc}"
                else:
                    item_str = f""
                
                if has_qty:
                    item_str += f" {qty}"
                
                if has_amt:
                    item_str += f" {amt}"
                
                line_items_parts.append(item_str)
        if not line_items_parts:
            warnings.warn(f"Transaction {txn_id}: no usable line item content, skipping.")
            continue

        line_items_str = " . ".join(line_items_parts)
        #line_items_str = " . ".join(line_items_parts) if line_items_parts else ""
                
        # Build the consolidated row
        new_row = {
            'Transaction ID': txn_id,
            'Invoice Date': first_row['Invoice Date'],
            'Tax': first_row['Tax'],
            'Currency': first_row['Currency'],
            'Line Items': line_items_str,
            'Vendor Name': first_row['Vendor Name'],
            'Label': first_row['Label'],
            'Account Code': first_row['Account Code'],
            'Price': first_row['Price']
        }
        
        consolidated_rows.append(new_row)
    
    # Create new dataframe
    result_df = pd.DataFrame(consolidated_rows)
    
    return result_df

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge transaction category and line item CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Expected file patterns in directory:
    {prefix}_categories.csv, {prefix}_categories1.csv, {prefix}_categories_1.csv, etc.
    {prefix}_line_items.csv, {prefix}_line_items1.csv, {prefix}_line_items_1.csv, etc.
        """,
    )
    parser.add_argument("prefix", help="File prefix (e.g., 'C1' or 'C2')")
    parser.add_argument("directory", help="Directory containing the CSV files")
    parser.add_argument(
        "-o", "--output",
        help="Output filename (default: {prefix}_merged.csv in the input directory)",
    )
    
    args = parser.parse_args()
    
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a valid directory.", file=sys.stderr)
        return 1
    
    prefix = args.prefix
    
    # Find files
    cat_files = find_files(directory, prefix, "_categories")
    lineitem_files = find_files(directory, prefix, "_line_items")
    
    print(f"Found {len(cat_files)} category file(s): {[f.name for f in cat_files]}")
    print(f"Found {len(lineitem_files)} line item file(s): {[f.name for f in lineitem_files]}")
    
    if not cat_files:
        print(f"Error: No {prefix}_categories*.csv files found in {directory}", file=sys.stderr)
        return 1
    if not lineitem_files:
        print(f"Error: No {prefix}_line_items*.csv files found in {directory}", file=sys.stderr)
        return 1
    
    # Load and concatenate
    print("\nLoading and concatenating files...")
    cat_df = load_and_concat(cat_files)
    lineitems_df = load_and_concat(lineitem_files)
    
    print(f"  Categories: {len(cat_df)} rows")
    print(f"  Line items: {len(lineitems_df)} rows")
    
    # Clean dataframes
    print("\nCleaning data...")
    cat_cleaned = clean_cat_df(cat_df)
    lineitems_cleaned = clean_lineitems_df(lineitems_df)
    
    print(f"  Categories after cleaning: {len(cat_cleaned)} rows")
    print(f"  Line items after cleaning: {len(lineitems_cleaned)} rows")
    
    # Merge
    print("\nMerging on 'Transaction ID'...")
    merged = lineitems_cleaned.merge(cat_cleaned, on="Transaction ID", how="left")

    merged.drop(columns =["Supplier", "Invoice Date_x", "Total", "Tax Amount"], inplace=True)

    # rename a column in merged from Invoice Date_x to Invoice Date
    merged = merged.rename(columns={'Invoice Date_y': 'Invoice Date'})

    print(f"  Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    
    # If there are some rows in teh merged dataset with null values in the Label column, print a warning
    if merged['Label'].isnull().any():
        warnings.warn("Warning: Some rows in the merged dataset have null values in the 'Label' column.")
    # hence remove rows with null values in the Label column and print the transaction IDs of the removed rows
        null_label_rows = merged[merged['Label'].isnull()]
        print(f"Removing {len(null_label_rows)} rows with null 'Label': Transaction IDs: {null_label_rows['Transaction ID'].tolist()}")
        merged = merged.dropna(subset=['Label'])
    
    

    
    consolidated = consolidate_transactions(merged)
    # drop the column currecy from both merged and consolidated
    merged = merged.drop(columns=['Currency'], errors='ignore')
    consolidated = consolidated.drop(columns=['Currency'], errors='ignore')
    consolidated_output_path = directory / f"{prefix}_consolidated.csv"
    consolidated.to_csv(consolidated_output_path, index=False)
    output_path = Path(args.output) if args.output else directory / f"{prefix}_merged.csv"
    merged.to_csv(output_path, index=False)

    print(f"\nSaved merged dataset to: {output_path}")

    print(f"Saved consolidated dataset to: {consolidated_output_path}")
     #sanity check the consolidated dataset has the same number of unique Transaction IDs 
     #as the cat_merged dataset 
    if consolidated['Transaction ID'].nunique() != cat_cleaned['Transaction ID'].nunique():
        warnings.warn("Warning: The number of unique Transaction IDs in the consolidated dataset does not match " \
        "the number in the category dataset. Something could have gone wrong in the scraping process.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
