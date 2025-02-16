import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate input data
def generate_input_data(sequence_length, num_samples):
    input_data = []
    for _ in range(num_samples):
        x = np.sin(np.arange(sequence_length)) + 0.2 * np.random.randn(sequence_length)
        input_data.append(x)
    return np.array(input_data)

sequence_length = 10
num_samples = 100
input_data = generate_input_data(sequence_length, num_samples)

# Convert input data to PyTorch tensor
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(2)

# Define RNN model wrapper
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 50, device=x.device)  # Ensure correct device
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])

# Initialize model
model = RNNModel()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(100):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, torch.tensor(input_data[:, -1], dtype=torch.float32).unsqueeze(1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# **Fix: Use `torch.jit.script` instead of `torch.jit.trace`**
scripted_model = torch.jit.script(model)

# Save scripted model
scripted_model.save("rnn_model.pt")

# Load scripted model
loaded_scripted_model = torch.jit.load("rnn_model.pt")

# Evaluate model (Disable gradients)
input_tensor_eval = torch.tensor(input_data, dtype=torch.float32).unsqueeze(2)
with torch.no_grad():
    predicted_outputs = loaded_scripted_model(input_tensor_eval).detach().numpy()

# Plot outputs
plt.figure(figsize=(10, 6))
plt.plot(input_data[:, -1], label='Actual')
plt.plot(predicted_outputs.flatten(), label='Predicted')
plt.legend()
plt.show()
