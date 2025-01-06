# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:21:15 2024

@author: Meshmesh
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Train a model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, "model.joblib")

# Load the model
loaded_model = joblib.load("model.joblib")

# Test the loaded model
print(loaded_model.predict(X[:5]))
