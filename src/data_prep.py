
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





processed_dir = PROJECT_ROOT / "data" / "processed"
if not processed_dir.exists():
    processed_dir.mkdir(exist_ok=True)

# Detect next version automatica
existing_versions = sorted(
    [int(f.stem.split("_")[0][1:]) for f in processed_dir.glob("v*_cleaned.csv")]
)

next_version = existing_versions[-1] + 1 if existing_versions else 1

new_filename = f"v{next_version}_cleaned.csv"
processed_path = processed_dir / new_filename

df.to_csv(processed_path, index=False)

print(f"Saved new processed dataset: {new_filename}")







