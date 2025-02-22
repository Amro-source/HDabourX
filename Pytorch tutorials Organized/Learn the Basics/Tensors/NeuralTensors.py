import torch
import torch.nn as nn
import torch.optim as optim

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

# Step 2: Prepare Dummy Data
torch.manual_seed(42)  # For reproducibility
X = torch.tensor([[0.5, 0.3], [0.1, 0.9], [0.7, 0.2], [0.8, 0.6]], dtype=torch.float32)  # Input features
y = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)  # Binary labels

print("Input Features (X):")
print(X)
print("\nTarget Labels (y):")
print(y)

# Step 3: Initialize the Model, Loss Function, and Optimizer
input_size = X.shape[1]  # Number of input features
hidden_size = 5  # Number of neurons in the hidden layer
output_size = 1  # Binary classification output

model = SimpleNN(input_size, hidden_size, output_size)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent

# Debugging: Print model parameters
print("\nModel Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param}")

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

print("\nPredicted Labels:")
print(predicted)
print("\nAccuracy:", accuracy.item())

# Debugging: Print final predictions and targets
print("\nFinal Predictions vs Targets:")
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {predicted[i]}")