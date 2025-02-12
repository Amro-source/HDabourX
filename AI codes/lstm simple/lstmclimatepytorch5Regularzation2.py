import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('DailyDelhiClimateTrain.csv')
df = df.drop('date', axis=1)  # Remove date column

# Define features and target
features = ['meantemp', 'humidity', 'wind_speed']
target = 'meanpressure'

# Normalize features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler(feature_range=(-1, 1))  # Center target around zero

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[target].values.reshape(-1, 1))

# Function to create sequences
def create_sequences(X, y, seq_length=50):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Create sequence data
seq_length = 50  # Increased sequence length
X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).squeeze()  # Ensure proper shape
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).squeeze()

# Define the improved LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=2, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # Only take the last time step

# Initialize model, optimizer, loss function, and learning rate scheduler
input_size = len(features)
model = LSTMModel(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)  # Lower LR + L2 Regularization
loss_fn = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)  # Step LR for stability

# Training loop with improved early stopping
num_epochs = 100
patience = 50
best_loss = float('inf')
counter = 0

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_torch).squeeze()  # Ensure shape matches y_train_torch
    loss = loss_fn(y_pred, y_train_torch)
    loss.backward()
    optimizer.step()

    # Adjust learning rate
    scheduler.step()

    # Early stopping check
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_torch).detach().cpu().numpy()  # Fix: Convert correctly

# Rescale predictions back to original range
y_test_pred_original = scaler_y.inverse_transform(y_test_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Apply final smoothing (moving average) for stability
y_test_pred_smoothed = np.convolve(y_test_pred_original.flatten(), np.ones(10)/10, mode='valid')

# Metrics
mse = mean_squared_error(y_test_original[:len(y_test_pred_smoothed)], y_test_pred_smoothed)
mae = mean_absolute_error(y_test_original[:len(y_test_pred_smoothed)], y_test_pred_smoothed)
r2 = r2_score(y_test_original[:len(y_test_pred_smoothed)], y_test_pred_smoothed)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R2: {r2:.2f}')

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test_original[:len(y_test_pred_smoothed)], label='Actual')
plt.plot(y_test_pred_smoothed, label='Predicted (Smoothed)')
plt.legend()
plt.show()
