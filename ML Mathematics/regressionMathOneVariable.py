import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. Generate Sample Data
# ======================
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # Random features (0 to 2)
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3X + noise

# ======================
# 2. Define Functions
# ======================
def hypothesis(X, theta0, theta1):
    """Predicts y using the linear model h(x) = theta0 + theta1 * X"""
    return theta0 + theta1 * X

def compute_cost(X, y, theta0, theta1):
    """Computes the Mean Squared Error (MSE) cost function"""
    m = len(X)
    predictions = hypothesis(X, theta0, theta1)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient_descent(X, y, theta0, theta1, learning_rate, iterations):
    """Performs gradient descent to optimize theta0 and theta1"""
    m = len(X)
    cost_history = []

    for _ in range(iterations):
        # Compute predictions
        predictions = hypothesis(X, theta0, theta1)

        # Compute gradients (partial derivatives)
        d_theta0 = (1/m) * np.sum(predictions - y)
        d_theta1 = (1/m) * np.sum((predictions - y) * X)

        # Update theta0 and theta1
        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1

        # Save cost for plotting
        cost = compute_cost(X, y, theta0, theta1)
        cost_history.append(cost)

    return theta0, theta1, cost_history

# ======================
# 3. Run Gradient Descent
# ======================
# Initialize parameters
theta0 = 0  # Initial bias
theta1 = 0  # Initial slope
learning_rate = 0.1
iterations = 100

# Run gradient descent
theta0_opt, theta1_opt, cost_history = gradient_descent(X, y, theta0, theta1, learning_rate, iterations)

print(f"Optimal theta0 (intercept): {theta0_opt:.2f}")
print(f"Optimal theta1 (slope): {theta1_opt:.2f}")

# ======================
# 4. Plot Results
# ======================
# Plot data and regression line
plt.figure(figsize=(12, 4))

# Plot 1: Data and best-fit line
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X, hypothesis(X, theta0_opt, theta1_opt), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()

# Plot 2: Cost function over iterations
plt.subplot(1, 2, 2)
plt.plot(range(iterations), cost_history, color='green')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function vs. Iterations')

plt.tight_layout()
plt.show()
