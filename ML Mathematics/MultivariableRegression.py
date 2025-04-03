import numpy as np

# ======================
# 1. Generate Sample Data
# ======================
np.random.seed(42)  # For reproducibility
m = 100  # Number of training examples
n = 3    # Number of features

# True parameters (θ₀ is intercept, θ₁-θ₃ are feature weights)
true_theta = np.array([4, 3, -2, 1])  # θ₀=4, θ₁=3, θ₂=-2, θ₃=1

# Generate random features (X) and compute y = Xθ + noise
X = np.random.rand(m, n) * 5  # Features in range [0, 5)
X_b = np.c_[np.ones((m, 1)), X]  # Add x₀=1 for intercept term (θ₀)
y = X_b.dot(true_theta) + np.random.randn(m) * 1.5  # Add Gaussian noise

# ======================
# 2. Define Key Functions (Matrix Form)
# ======================
def compute_cost(X, y, theta):
    """Compute MSE cost function (matrix form)"""
    m = len(y)
    errors = X.dot(theta) - y
    return (1/(2*m)) * errors.T.dot(errors)  # (Xθ - y)ᵀ(Xθ - y)

def gradient_descent(X, y, theta, alpha, iterations):
    """Perform gradient descent (matrix form)"""
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)  # Xᵀ(Xθ - y)
        theta -= alpha * gradients
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# ======================
# 3. Run Gradient Descent
# ======================
# Initialize parameters (θ₀, θ₁, θ₂, θ₃)
theta_init = np.random.randn(n + 1)  # Random initialization
alpha = 0.1  # Learning rate
iterations = 500

theta_opt, cost_history = gradient_descent(X_b, y, theta_init, alpha, iterations)

print("Optimized parameters:")
print(f"θ₀ (Intercept): {theta_opt[0]:.4f} (True: {true_theta[0]})")
print(f"θ₁ (Feature 1): {theta_opt[1]:.4f} (True: {true_theta[1]})")
print(f"θ₂ (Feature 2): {theta_opt[2]:.4f} (True: {true_theta[2]})")
print(f"θ₃ (Feature 3): {theta_opt[3]:.4f} (True: {true_theta[3]})")

# ======================
# 4. Plot Cost vs. Iterations
# ======================
import matplotlib.pyplot as plt
plt.plot(range(iterations), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Over Iterations")
plt.show()

# ======================
# 5. Compare with Normal Equation
# ======================
# Analytical solution (θ = (XᵀX)⁻¹Xᵀy)
theta_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("\nParameters from Normal Equation:")
print(theta_normal)