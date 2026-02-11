# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 11:21:11 2026

@author: Rohit
"""

import requests

URL = "http://127.0.0.1:8000/predict"

# Sample valid input (Type must be string!)
payload = {
    "features": ["L", 300, 310, 1500, 40, 0]
}

# -------------------------
# Test 1: API reachable
# -------------------------
response = requests.post(URL, json=payload)
assert response.status_code == 200
print("Test 1 passed: API reachable")

# -------------------------
# Test 2: Response format
# -------------------------
data = response.json()
assert "predictions" in data
print("Test 2 passed: Response contains predictions")



# -------------------------
# Test 3: Output shape
# -------------------------
preds = data["predictions"]
print("Predictions:", preds)
print("Number of outputs:", len(preds[0]))


assert isinstance(preds, list) # checks if the returned variable is a list type
assert len(preds[0]) == 6   # 6 target variables
print("Test 3 passed: Output shape correct")
