# Import necessary libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(0)

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

# Reshape data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# Make predictions
predictions = model.predict(X_test)

# Plot results
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
