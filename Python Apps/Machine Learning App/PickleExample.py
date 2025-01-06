# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 15:20:06 2024

@author: Meshmesh
"""

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Train a model
data = load_iris()
X, y = data.data, data.target
model = LogisticRegression()
model.fit(X, y)

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Load the model
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Test the loaded model
print(loaded_model.predict(X[:5]))
