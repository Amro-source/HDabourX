import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate sample data
def generate_data():
    t = np.arange(0, 100, 0.1)
    x = np.sin(t) + 0.2 * np.random.randn(len(t))
    return t, x

# Generate training and testing data
t_train, x_train = generate_data()
t_test, x_test = generate_data()

# Reshape data for LSTM input
def reshape_data(data, sequence_length):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
X_train, y_train = reshape_data(x_train, sequence_length)
X_test, y_test = reshape_data(x_test, sequence_length)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define custom dataset class
class LSTMDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

# Create data loaders
train_dataset = LSTMDataSet(X_train, y_train)
test_dataset = LSTMDataSet(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out

# Initialize model, optimizer, and loss function
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(100):
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate model
model.eval()
with torch.no_grad():
    total_loss = 0
    for X, y in test_loader:
        outputs = model(X)
        loss = criterion(outputs, y)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss / len(test_loader)}')

# Use tracing to convert model to TorchScript
input_tensor = torch.randn(1, sequence_length, 1)
traced_script_module = torch.jit.trace(model, input_tensor)
traced_script_module.save("lstm_model.pt")

# Load TorchScript model and evaluate
loaded_script_module = torch.jit.load("lstm_model.pt")
loaded_script_module.eval()
with torch.no_grad():
    total_loss = 0
    for X, y in test_loader:
        outputs = loaded_script_module(X)
        loss = criterion(outputs, y)
        total_loss += loss.item()
    print(f'Test Loss (TorchScript): {total_loss / len(test_loader)}')

# Plot results
model.eval()
with torch.no_grad():
    predictions = []
    for X, _ in test_loader:
        outputs = model(X)
        predictions.extend(outputs.squeeze(1).numpy())

plt.plot(y_test.squeeze(1).numpy(), label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
