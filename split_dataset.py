# split_clients.py
import pandas as pd
import numpy as np
import os

NUM_CLIENTS = 2
DATA_PATH = "/home/kaifalam/kaif/SSDS/fraud_detection/data/creditcard.csv"
OUT_DIR = "data"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
splits = np.array_split(df, NUM_CLIENTS)

for i, split_df in enumerate(splits):
    split_df.to_csv(f"{OUT_DIR}/client{i}.csv", index=False)
    print(f"Saved: client{i}.csv with {len(split_df)} rows")
