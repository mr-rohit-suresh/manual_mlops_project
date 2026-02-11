
import pandas as pd
import yaml
from pathlib import Path



# =========================
# PROJECT ROOT
# =========================
#try:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print('project root is ', PROJECT_ROOT)
#except NameError:
#    PROJECT_ROOT = Path.cwd()

# =========================
# LOAD CONFIG
# =========================
config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# =========================
# LOAD RAW DATA
# =========================
#raw_path = PROJECT_ROOT / "data" / "raw" / "v1_raw.csv"
raw_path=PROJECT_ROOT /cfg['data']['raw_path']
df = pd.read_csv(raw_path)

# =========================
# CLEANING STEPS
# =========================


cols_to_drop = cfg['data']['cols_to_drop']
df = df.drop(columns=cols_to_drop)
#df = df.drop(columns=cols_to_drop, errors="ignore")



processed_path = PROJECT_ROOT / "data" / "processed" / "v2_cleaned.csv"
df.to_csv(processed_path, index=False)

print("Cleaned data saved to:", processed_path)
