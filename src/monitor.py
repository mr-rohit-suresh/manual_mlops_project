# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 12:34:00 2026

@author: Rohit
"""

import pandas as pd
from pathlib import Path
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)
    
path=PROJECT_ROOT / cfg['data']['processed_dir']



processed_dir = PROJECT_ROOT / cfg['data']['processed_dir']


# Detect next version automatica
existing_versions = sorted(
    [int(f.stem.split("_")[0][1:]) for f in processed_dir.glob("v*_cleaned.csv")]
)

next_version = existing_versions[-1]  if existing_versions else 1
new_filename = processed_dir / f"v{next_version}_cleaned.csv"

print (f'used {new_filename} for modifying the data')
#new_filename = f"production.csv"
df = pd.read_csv(PROJECT_ROOT / new_filename)
test_samp=len(df)-cfg['deployment']['test_samp']


# Simulate later time period
day2 = df.iloc[test_samp:].copy()

# Simulate drift (small realistic changes)
day2["Air temperature [K]"] += np.random.normal(1, 0.5*2, len(day2))
day2["Torque [Nm]"] += np.random.normal(2, 1.0*2, len(day2))

path=PROJECT_ROOT / cfg['deployment']['data_dir']
if not path.exists():
    path.mkdir()
new_path= path/ f"production_sample_test.csv"
day2.to_csv(new_path, index=False)
