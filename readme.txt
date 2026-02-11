1. v1_raw.csv is the raw data file. run  python /manual_mlops_project/src/data_prep.py to clean the data. This will remove the unwanted column basically. The cleaned data will be available in the data folder in the name of v2_cleaned.csv.

2. run python  /manual_mlops_project/src/train.py. to train the model using the cleaned data (v2_cleaned.csv). The hyperparameters can be modified in the the config.yaml file.

3. The trained model is available for testing in the URL = "http://127.0.0.1:8000/predict". run /manual_mlops_project/src/inference.py. The input query need to be of the form "features": ["L", 298.1, 308.6, 1551, 42.8, 0]. where the inputs reads the values in the respective order of "Type",  "Air temperature [K]",    "Process temperature [K]",    "Rotational speed [rpm]",    "Torque [Nm]",    "Tool wear [min]". 

4. The trained model can also be tested through python scripts without directly accessing the url. run /manual_mlops_project/src/smoke_tests.py by modifygin the sample test data. The outcome of the test will reveal if the connection to URL was error free. This doesnt indicate the performance of the model.

uvicorn src.inference:app --reload

5. The performance of the model can be tested by feeding a noisy data. run /manual_mlops_project/src/monitor.py to modify few columns in the existing data. The new test data will be saved as v3_day2.csv. run /manual_mlops_project/src/run_day2.py to compare the accuracy of the prediction. If the accuracy drops below the threshold, the model can be retrained. 