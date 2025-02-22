import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1: Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer 1
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Fully connected layer 2
        self.sigmoid = nn.Sigmoid()  # Output activation function

    def forward(self, x):
        # Forward pass
        out = self.fc1(x)  # Linear transformation
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Linear transformation
        out = self.sigmoid(out)  # Apply sigmoid activation
        return out


# Step 2: Generate Dummy Data with 100 Samples
torch.manual_seed(42)  # For reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate random input features (2-dimensional)
X = torch.tensor(np.random.rand(num_samples, 2), dtype=torch.float32)

# Generate random binary labels (0 or 1)
y = torch.tensor(np.random.randint(0, 2, size=(num_samples, 1)), dtype=torch.float32)

print("Input Features (X):")
print(X[:5])  # Print the first 5 samples for inspection
print("\nTarget Labels (y):")
print(y[:5])  # Print the first 5 labels for inspection

# Step 3: Initialize the Model, Loss Function, and Optimizer
input_size = X.shape[1]  # Number of input features
hidden_size = 5  # Number of neurons in the hidden layer
output_size = 1  # Binary classification output

model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Step 4: Train the Neural Network
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X)  # Pass input through the model
    loss = criterion(predictions, y)  # Compute loss

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Make Predictions and Evaluate Accuracy
with torch.no_grad():  # Disable gradient computation during inference
    predicted = model(X).round()  # Round the output to get binary predictions
    accuracy = (predicted == y).float().mean()  # Calculate accuracy

print("\nPredicted Labels (First 5 Samples):")
print(predicted[:5])
print("\nAccuracy:", accuracy.item())

# Debugging: Print final predictions vs targets for the first 5 samples
print("\nFinal Predictions vs Targets (First 5 Samples):")
for i in range(5):
    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {predicted[i]}")