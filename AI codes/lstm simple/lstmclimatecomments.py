# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
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

# Reshape the data for LSTM
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, X_train_scaled.shape[2])))  # Add LSTM layer with 50 units
model.add(Dense(1))  # Add output layer with 1 unit
model.compile(loss='mean_squared_error', optimizer='adam')  # Compile the model with MSE loss and Adam optimizer

# Train the model
model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=1)  # Train the model for 100 epochs with batch size 32

# Make predictions
y_pred_scaled = model.predict(X_test_scaled)  # Make predictions on the test set

# Invert the scaling to get the original output values
y_pred = y_scaler.inverse_transform(y_pred_scaled)  # Invert the scaling to get the original output values

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Calculate MSE
mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
r2 = r2_score(y_test, y_pred)  # Calculate R2 score

print(f'MSE: {mse:.2f}')  # Print MSE
print(f'MAE: {mae:.2f}')  # Print MAE
print(f'R2: {r2:.2f}')  # Print R2 score

# Plot the predicted values against the actual values
plt.plot(y_test.values, label='Actual')  # Plot actual values
plt.plot(y_pred, label='Predicted')  # Plot predicted values
plt.legend()  # Add legend
plt.show()  # Show the plot

