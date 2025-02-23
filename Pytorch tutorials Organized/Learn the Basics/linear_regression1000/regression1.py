import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Generate synthetic dataset with 1000 samples
np.random.seed(42)  # For reproducibility
x_train = np.random.rand(1000, 1).astype(np.float32) * 10  # Random values between 0 and 10
y_train = 2 * x_train + 1 + np.random.randn(1000, 1).astype(np.float32) * 1.5  # Linear relationship with noise

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()

plt.figure(figsize=(8, 6))
plt.scatter(x_train, y_train, color='red', label='Original data', alpha=0.5)
plt.plot(x_train, predicted, color='blue', label='Fitted line')
plt.title('Linear Regression with 1000 Samples')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')