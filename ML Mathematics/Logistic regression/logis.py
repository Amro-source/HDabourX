import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, lambda_=0.1, regularization='l2', verbose=False):
        """
        Parameters:
        - learning_rate: Step size for gradient descent
        - n_iter: Number of iterations
        - lambda_: Regularization strength
        - regularization: 'l1' or 'l2' regularization
        - verbose: Whether to print progress
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.regularization = regularization.lower()
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X, y):
        """Compute the cost function with regularization"""
        m = X.shape[0]
        z = np.dot(X, self.weights) + self.bias
        h = self._sigmoid(z)
        
        # Cross-entropy loss
        cost = (-1/m) * np.sum(y * np.log(h + 1e-10) + (1-y) * np.log(1-h + 1e-10))
        
        # Add regularization
        if self.regularization == 'l2':
            reg_term = (self.lambda_/(2*m)) * np.sum(self.weights**2)
        elif self.regularization == 'l1':
            reg_term = (self.lambda_/m) * np.sum(np.abs(self.weights))
        else:
            reg_term = 0
            
        return cost + reg_term
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        
        for i in range(self.n_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            h = self._sigmoid(z)
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (h - y))
            db = (1/m) * np.sum(h - y)
            
            # Add regularization to gradient
            if self.regularization == 'l2':
                dw += (self.lambda_/m) * self.weights
            elif self.regularization == 'l1':
                dw += (self.lambda_/m) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Print progress
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
                
        return self
    
    def predict_proba(self, X):
        """Return probability estimates"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Return class predictions"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def plot_cost_history(self):
        """Plot the cost function over iterations"""
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.show()

# Example usage with synthetic data
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                             n_redundant=2, random_state=42)
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model with L2 regularization
    print("Training with L2 regularization...")
    model_l2 = LogisticRegression(learning_rate=0.1, n_iter=2000, lambda_=0.5, 
                                regularization='l2', verbose=True)
    model_l2.fit(X_train, y_train)
    model_l2.plot_cost_history()
    
    # Evaluate
    y_pred = model_l2.predict(X_test)
    print(f"Test Accuracy (L2): {accuracy_score(y_test, y_pred):.4f}")
    
    # Train model with L1 regularization
    print("\nTraining with L1 regularization...")
    model_l1 = LogisticRegression(learning_rate=0.1, n_iter=2000, lambda_=0.5, 
                                regularization='l1', verbose=True)
    model_l1.fit(X_train, y_train)
    model_l1.plot_cost_history()
    
    # Evaluate
    y_pred = model_l1.predict(X_test)
    print(f"Test Accuracy (L1): {accuracy_score(y_test, y_pred):.4f}")
    
    # Compare coefficients
    print("\nCoefficient comparison:")
    print(f"L2 - Non-zero coefficients: {(np.abs(model_l2.weights) > 0.01).sum()}/{len(model_l2.weights)}")
    print(f"L1 - Non-zero coefficients: {(np.abs(model_l1.weights) > 0.01).sum()}/{len(model_l1.weights)}")