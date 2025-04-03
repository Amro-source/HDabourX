import numpy as np

# Single neuron example
inputs = np.array([0.5, 0.3])
weights = np.array([1.2, -0.4])
bias = 0.1

# Forward pass
z = np.dot(inputs, weights) + bias
output = 1 / (1 + np.exp(-z))  # sigmoid

# Backward pass (true label = 0)
error = output - 0
gradient = error * output * (1 - output)  # sigmoid derivative
weight_updates = gradient * inputs
new_weights = weights - 0.1 * weight_updates  # learning rate=0.1