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

df = pd.read_csv(PROJECT_ROOT / path)
test_samp=len(df)-cfg['test_samp']


# Simulate later time period
day2 = df.iloc[test_samp:].copy()

# Simulate drift (small realistic changes)
day2["Air temperature [K]"] += np.random.normal(1, 0.5*2, len(day2))
day2["Torque [Nm]"] += np.random.normal(2, 1.0*2, len(day2))

path=PROJECT_ROOT / cfg['deployment']['data_dir']
day2.to_csv(path, index=False)
