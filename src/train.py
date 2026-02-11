# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 16:10:59 2026

@author: Rohit
"""

import yaml
from pathlib import Path
import pickle
from datetime import datetime
#root=Path.cwd().resolve().parent.parent
#config_path=root/"Assignments/config/config.yaml"
#data_path=root/'Assignments/'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
config_path = PROJECT_ROOT / "config.yaml"

# =========================
# LOAD CONFIG
# =========================
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

#csv_path=data_path/cfg['data']['input_csv']
csv_path = PROJECT_ROOT / cfg["data"]["processed_dir"]

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score



# Load data
data = pd.read_csv(csv_path)
nf=cfg['data']['no_of_features']
X = data.iloc[:, :nf]

# Columns 9–14 → targets
y = data.iloc[:, nf:].values


train_size=cfg['model_params']['train_size']
X_train = X.iloc[:train_size]
X_test  = X.iloc[train_size:]

y_train = y[:train_size]
y_test  = y[train_size:]

# =========================
# PREPROCESSING
# =========================
# Feature 1 (index 0 in X) → categorical
categorical_features = [cfg['preprocessing']['categorical_feature_index']]

# Remaining features → numeric
numeric_features = cfg['preprocessing']['numeric_feature_indices']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ]
)

# =========================
# MODEL
# =========================
base_rf = RandomForestClassifier(
    n_estimators=cfg["model_params"]["n_estimators"],
    max_depth=cfg["model_params"].get("max_depth", None),
    random_state=cfg["model_params"]["random_state"],
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", MultiOutputClassifier(base_rf))
])

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
y_pred = model.predict(X_test)



# =========================
# EVALUATION
# =========================
exact_match_accuracy = np.mean(np.all(y_pred == y_test, axis=1))
print("Exact match accuracy:", exact_match_accuracy)

for i in range(y_test.shape[1]):
    acc = accuracy_score(y_test[:, i], y_pred[:, i])
    print(f"Target {i+1} accuracy: {acc:.4f}")



# Create artifacts directories
model_dir = PROJECT_ROOT / cfg["artifacts"]["model_dir"]
model_dir.mkdir(parents=True, exist_ok=True)

# Model versioning
model_version = "model_v1.pkl"
model_path = model_dir / model_version


# Save model
with open(model_path, "wb") as f:
    pickle.dump(model, f)



# Create artifacts directories
model_dir = PROJECT_ROOT / cfg["deployment"]["model_dir"]
model_dir.mkdir(parents=True, exist_ok=True)

# Model versioning
model_version = "model_v1.pkl"
model_path = model_dir / model_version


# Save model
with open(model_path, "wb") as f:
    pickle.dump(model, f)



print(f"Model saved at: {model_path}")

# =========================
# SAVE METADATA
# =========================
import json
import subprocess
from datetime import datetime

# Create metadata directory
metadata_path = PROJECT_ROOT / cfg["deployment"]["metadata_path"]
#metadata_dir.mkdir(parents=True, exist_ok=True)

# Get git commit hash (safe fallback)
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT
    ).decode("utf-8").strip()
except Exception:
    git_commit = "not_a_git_repo"

metadata = {
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_version": cfg["data"]["processed_dir"],
    "train_size": cfg["model_params"]["train_size"],
    "model_type": cfg['model_params']['algorithm'],
    #"solver": cfg["model"]["solver"],
    "max_iter": cfg["model_params"]["max_iter"],
    "exact_match_accuracy": exact_match_accuracy,
    "git_commit": git_commit,
    "config_used": "config.yaml",
    "model_file": str(model_path.name)
}

#metadata_path = metadata_dir / "model_v1_metadata.json"

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved at: {metadata_path}")
