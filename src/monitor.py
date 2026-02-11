# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 12:34:00 2026

@author: Rohit
"""

import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(PROJECT_ROOT / "data" /"processed"/ "v2_cleaned.csv")

# Simulate later time period
day2 = df.iloc[3000:].copy()

# Simulate drift (small realistic changes)
day2["Air temperature [K]"] += np.random.normal(1, 0.5*2, len(day2))
day2["Torque [Nm]"] += np.random.normal(2, 1.0*2, len(day2))

day2.to_csv(PROJECT_ROOT / "data"/"processed" / "v3_day2.csv", index=False)
