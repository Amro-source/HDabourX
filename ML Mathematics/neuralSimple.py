import numpy as np

# Forward pass
z1 = W1.dot(x) + b1
h = sigmoid(z1)
z2 = W2.dot(h) + b2
y_hat = sigmoid(z2)

# Loss
J = 0.5 * np.sum((y_hat - y)**2)

# Backpropagation
dJ_dy_hat = y_hat - y
dJ_dz2 = dJ_dy_hat * sigmoid_derivative(z2)
dJ_dW2 = dJ_dz2.dot(h.T)
dJ_db2 = dJ_dz2

dJ_dh = W2.T.dot(dJ_dz2)
dJ_dz1 = dJ_dh * sigmoid_derivative(z1)
dJ_dW1 = dJ_dz1.dot(x.T)
dJ_db1 = dJ_dz1

# Gradient descent
W1 -= learning_rate * dJ_dW1
b1 -= learning_rate * dJ_db1
W2 -= learning_rate * dJ_dW2
b2 -= learning_rate * dJ_db2