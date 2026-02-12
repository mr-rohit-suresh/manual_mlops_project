# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 12:34:57 2026

@author: Rohit
"""

import pandas as pd
import requests
import yaml
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent

config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)
    
label_errors = 0
total_labels = 0


URL = "http://127.0.0.1:8000/predict"
path=PROJECT_ROOT / cfg['deployment']['data_dir']
df = pd.read_csv(path)
#df = pd.read_csv("data/processed/v3_day2.csv")

errors = 0
total = len(df)

for _, row in df.iterrows():
    payload = {"features": row.iloc[:6].tolist()}
    r = requests.post(URL, json=payload)

    pred = [int(x) for x in r.json()["predictions"][0]]
    true = row.iloc[6:12].astype(int).tolist()

    for p, t in zip(pred, true):
        total_labels += 1
        if p != t:
            label_errors += 1

error_rate = label_errors / total_labels
print("Production error rate:", error_rate*100, "%")


if error_rate > cfg['deployment']['threshold']:
    print("retrain the model")
