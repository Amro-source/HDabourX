import numpy as np

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize parameters
x = 2          # Input
y = 1          # True output
w1 = 0.5       # Input -> hidden weight
b1 = 0.1       # Hidden bias
w2 = 0.3       # Hidden -> output weight
b2 = 0.05      # Output bias
learning_rate = 0.1

# Forward pass
z1 = w1 * x + b1            # Eq. 1
h = sigmoid(z1)             # Eq. 2
z2 = w2 * h + b2            # Eq. 3
y_hat = sigmoid(z2)         # Eq. 4

# Loss calculation
loss = 0.5 * (y_hat - y)**2 # Eq. 5

# Backpropagation
# Output layer gradients
dJ_dy_hat = y_hat - y                   # Eq. 6
dy_hat_dz2 = sigmoid_derivative(y_hat)   # Eq. 7
dJ_dz2 = dJ_dy_hat * dy_hat_dz2          # Eq. 8

dJ_dw2 = dJ_dz2 * h                      # Eq. 9
dJ_db2 = dJ_dz2                          # Eq. 10

# Hidden layer gradients
dJ_dh = dJ_dz2 * w2                      # Eq. 11
dh_dz1 = sigmoid_derivative(h)           # Eq. 12
dJ_dz1 = dJ_dh * dh_dz1                  # Eq. 13

dJ_dw1 = dJ_dz1 * x                      # Eq. 14
dJ_db1 = dJ_dz1                          # Eq. 15

# Update weights (gradient descent)
w1 -= learning_rate * dJ_dw1
b1 -= learning_rate * dJ_db1
w2 -= learning_rate * dJ_dw2
b2 -= learning_rate * dJ_db2

# Print results
print("Forward pass:")
print(f"z1 = {z1:.4f}, h = {h:.4f}")
print(f"z2 = {z2:.4f}, y_hat = {y_hat:.4f}")
print(f"Loss = {loss:.4f}\n")

print("Gradients:")
print(f"dJ/dw2 = {dJ_dw2:.6f}, dJ/db2 = {dJ_db2:.6f}")
print(f"dJ/dw1 = {dJ_dw1:.6f}, dJ/db1 = {dJ_db1:.6f}\n")

print("Updated weights:")
print(f"New w1 = {w1:.6f}, new b1 = {b1:.6f}")
print(f"New w2 = {w2:.6f}, new b2 = {b2:.6f}")