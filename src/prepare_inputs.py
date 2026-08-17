'''
Create the three different input formats for the model:
Input A: "Vendor Name, Invoice Date, Price"
Input B: "Vendor Name, Invoice Date, Line Items, Price"
Input C: "Vendor Name: {Vendor Name}, Invoice Date: {Invoice Date}, Line Items: {Line Items}, Price: {Price}"   
'''
import pandas as pd
import argparse
import sys
from pathlib import Path
from config import consolidated_csv, input_csv

def create_input_A(df):
    df = df.copy()
    if 'Vendor Name' not in df.columns or 'Invoice Date' not in df.columns or 'Price' not in df.columns:
        raise ValueError("One or more required columns are missing from the dataframe.")
    # if either of the columns have missing values, fill them with an empty string
    df['Vendor Name'] = df['Vendor Name'].fillna('')
    df['Invoice Date'] = df['Invoice Date'].fillna('')
    df['Price'] = df['Price'].fillna('')
    df['Text Input'] = df['Vendor Name'] + ', ' + df['Invoice Date'].astype(str) + ', ' + df['Price'].astype(str)
    return df[['Transaction ID', 'Text Input', 'Label']]

def create_input_B(df):
    df = df.copy()
    if 'Vendor Name' not in df.columns or 'Invoice Date' not in df.columns or 'Price' not in df.columns or 'Line Items' not in df.columns:
        raise ValueError("One or more required columns are missing from the dataframe.")
    # if either of the columns have missing values, fill them with an empty string
    df['Vendor Name'] = df['Vendor Name'].fillna('')
    df['Invoice Date'] = df['Invoice Date'].fillna('')
    df['Price'] = df['Price'].fillna('')
    df['Line Items'] = df['Line Items'].fillna('')
    df['Text Input'] = df['Vendor Name'] + ', ' + df['Invoice Date'].astype(str) + ', '  + df['Line Items'].astype(str) + ', ' + df['Price'].astype(str)


    return df[['Transaction ID', 'Text Input', 'Label']]

def create_input_C(df):
    df = df.copy()
    if 'Vendor Name' not in df.columns or 'Invoice Date' not in df.columns or 'Price' not in df.columns or 'Line Items' not in df.columns:
        raise ValueError("One or more required columns are missing from the dataframe.")
    # if either of the columns have missing values, fill them with an empty string
    df['Vendor Name'] = df['Vendor Name'].fillna('')
    df['Invoice Date'] = df['Invoice Date'].fillna('')
    df['Price'] = df['Price'].fillna('')
    df['Line Items'] = df['Line Items'].fillna('')
    df['Text Input'] = 'Vendor Name: ' + df['Vendor Name'] + ', Invoice Date: ' + df['Invoice Date'].astype(str) + ', Line Items: ' + df['Line Items'].astype(str) + ', Price: ' + df['Price'].astype(str)


    return df[['Transaction ID', 'Text Input', 'Label']]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the three different input formats for the model.",
    )
    parser.add_argument("prefix", help="Client code (e.g., 'C1' or 'C2')")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
 
    
    prefix = args.prefix
 
    # Load
    csv_path = consolidated_csv(prefix)
    print(F"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
 
    # Create the three input formats
    creators = {"A": create_input_A, "B": create_input_B, "C": create_input_C}
    for input_type, create_fn in creators.items():
        out_path = input_csv(prefix, input_type)
        create_fn(df).to_csv(out_path, index=False)
        print(f"Input {input_type} saved to: {out_path}")
 
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())
 
