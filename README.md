# Manual MLOps Pipeline – Machine Failure Prediction

## Project Overview

The objective of this project is to predict **machine failure** using sensor and operational data.  
The model uses the following input features:

- Type  
- Air temperature [K]  
- Process temperature [K]  
- Rotational speed [rpm]  
- Torque [Nm]  
- Tool wear [min]  

This project demonstrates a **manual MLOps pipeline**, covering data versioning, training, deployment, monitoring, and retraining.


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/mr-rohit-suresh/manual_mlops_project.git
cd manual_mlops_project

### 2. Instal dependencies
 ```bash
pip install -r requirements.txt


Phase A: Data Preparation

v1_raw.csv is the original raw dataset. Run the data cleaning script:
```bash 
python src/data_prep.py

Train the model using the cleaned dataset.

```bash
python src/train.py
Training configuration and hyperparameters are controlled via config.yaml

Start the API Server
```bash
uvicorn src.inference:app --reload

API will be available at:
Swagger UI: http://127.0.0.1:8000/docs

Prediction endpoint: POST /predict

Sample Input Format
```bash 
{
  "features": ["L", 298.1, 308.6, 1551, 42.8, 0]
}

Phase C: Smoke Testing

To verify that the API is running correctly. Open Terminal 1 and start the API:

```bash
uvicorn src.inference:app --reload


Open Terminal 2 and run:

```bash
python tests/smoke_tests.py


These tests confirm API connectivity and correct response format. They do not measure model accuracy.

Phase D: Monitoring & Drift Simulation
Simulate Data Drift

```bash
python src/monitor.py


Generates a drifted dataset:

```bash
data/processed/v3_day2.csv

Evaluate Production Performance
```bash
python tests/run_day2.py


This compares predictions against ground truth and computes production error rate. If error exceeds the threshold, retraining is recommended