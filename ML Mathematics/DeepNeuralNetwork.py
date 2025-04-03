import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = [np.random.randn(n, m) * 0.01 
                       for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((n, 1)) for n in layer_sizes[1:]]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def forward(self, x):
        a = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b  # Eq. 1
            a = self.sigmoid(z)    # Eq. 2
            zs.append(z)
            activations.append(a)
        return activations, zs
    
    def backward(self, x, y, activations, zs):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        
        # Output layer error
        delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])  # Eq. 3
        nabla_w[-1] = np.dot(delta, activations[-2].T)              # Eq. 5
        nabla_b[-1] = delta                                         # Eq. 6
        
        # Hidden layers
        for l in range(2, len(self.weights)+1):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * self.sigmoid_prime(z)  # Eq. 4
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            nabla_b[-l] = delta
        return nabla_w, nabla_b
    
    def update(self, nabla_w, nabla_b, lr):
        self.weights = [w - lr * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - lr * nb for b, nb in zip(self.biases, nabla_b)]
    
    def train(self, X, Y, epochs, lr):
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                # Reshape inputs
                x = x.reshape(-1, 1)
                y = y.reshape(-1, 1)
                
                # Forward pass
                activations, zs = self.forward(x)
                
                # Backward pass
                nabla_w, nabla_b = self.backward(x, y, activations, zs)
                
                # Update weights
                self.update(nabla_w, nabla_b, lr)

# Example usage
layer_sizes = [2, 4, 3, 1]  # 2 inputs, 2 hidden layers (4 and 3 neurons), 1 output
dnn = DeepNeuralNetwork(layer_sizes)

# Sample data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
dnn.train(X, Y, epochs=1000, lr=0.1)