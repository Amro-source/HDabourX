# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('DailyDelhiClimateTrain.csv')

# Drop the date column as it's not relevant for this problem
df = df.drop('date', axis=1)

# Define the feature columns
feature_cols = ['meantemp', 'humidity', 'wind_speed']

# Define the target column
target_col = 'meanpressure'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.2, random_state=42)

# Create a scaler object for the input values
scaler = MinMaxScaler()

# Fit the scaler to the training input values and transform both the training and testing input values
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a scaler object for the output values
y_scaler = MinMaxScaler()

# Fit the scaler to the training output values and transform both the training and testing output values
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

# Convert the data to PyTorch tensors
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).squeeze(1)
y_test_scaled = torch.tensor(y_test_scaled, dtype=torch.float32).squeeze(1)

# Define the LSTM model
lstm = nn.LSTM(input_size=3, hidden_size=50, num_layers=1, batch_first=True)
dropout = nn.Dropout(p=0.2)
fc = nn.Linear(50, 1)

# Initialize the optimizer and loss function
optimizer = optim.Adam(list(lstm.parameters()) + list(dropout.parameters()) + list(fc.parameters()), lr=0.001, weight_decay=0.01)
loss_fn = nn.MSELoss()

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    out, _ = lstm(X_train_scaled)
    out = dropout(out[:, -1, :])
    out = fc(out)
    loss = loss_fn(out, y_train_scaled)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Make predictions
out, _ = lstm(X_test_scaled)
out = dropout(out[:, -1, :])
out = fc(out)

# Invert the scaling to get the original output values
y_pred = y_scaler.inverse_transform(out.detach().numpy())

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R2: {r2:.2f}')

# Plot the predicted values against the actual values
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
