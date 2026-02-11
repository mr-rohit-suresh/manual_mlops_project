from fastapi import FastAPI
import pickle
import pandas as pd
from pathlib import Path
import csv

FEATURE_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]



app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts/models/model_v1.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("Model expects features:", model.feature_names_in_)


# when ever a button is clicked in the webpage, HTTP post request is trigerred
# Once that happens, the function beneth this is trigerred.
# The base url is created starting with 127. This will command the system to skip 
# the network interface card and  loop back to the same local system and pass ntg 
# to the internet. Nobody can thefore access this url created by us in our local system
# in their system


@app.post("/predict")
def predict(payload: dict):
    X = pd.DataFrame(
        [payload["features"]],
        columns=FEATURE_COLUMNS
    )
    preds = model.predict(X)
    print('app.py', preds)
    return {"predictions": preds.tolist()}


from datetime import datetime
import csv

DEPLOYMENT_LOG = PROJECT_ROOT / "deployment_log.csv"

with open(DEPLOYMENT_LOG, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        datetime.now().isoformat(),
        "v1",
        str(MODEL_PATH),
        "manual deployment"
    ])



