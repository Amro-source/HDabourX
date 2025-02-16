import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the saved LSTM model
loaded_script_module = torch.jit.load("lstm_model.pt")
loaded_script_module.eval()

# Generate some input data
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

# Get model outputs
with torch.no_grad():
    outputs = loaded_script_module(input_tensor)

# Plot outputs
plt.figure(figsize=(10, 6))
plt.plot(input_data.flatten()[:len(outputs.numpy().flatten())], label='Actual')
plt.plot(outputs.numpy().flatten(), label='Predicted')
plt.legend()
plt.show()
