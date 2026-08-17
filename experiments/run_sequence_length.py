#%%
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import input_csv, CLIENTS, INPUT_TYPES

from transformers import AutoTokenizer
#%%
checkpoint= "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#%%
for client in CLIENTS:
    for input_type in INPUT_TYPES:
        df = pd.read_csv(input_csv(client, input_type))
        sentences = df["Text Input"].tolist()
        
        sentences_tokenized = tokenizer(sentences)
        lengths = [len(ids) for ids in sentences_tokenized["input_ids"]]
        num_long = np.sum(np.array(lengths) > 512)
        print(f"{client} {input_type}: {num_long} sentences exceed 512 tokens")

# %%
