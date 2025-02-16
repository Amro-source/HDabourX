
# Import necessary libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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

# Define different LSTM architectures
def lstm_architecture_1():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def lstm_architecture_2():
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def lstm_architecture_3():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Train and evaluate different LSTM architectures
architectures = [lstm_architecture_1, lstm_architecture_2, lstm_architecture_3]
predictions = []
mses = []

for i, architecture in enumerate(architectures):
    model = architecture()
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)
    prediction = model.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    predictions.append(prediction)
    mses.append(mse)
    print(f'Architecture {i+1} MSE: {mse}')

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', color='black')
for i, prediction in enumerate(predictions):
    plt.plot(prediction, label=f'Architecture {i+1}')
plt.legend()
plt.title('Comparison of LSTM Architectures')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.show()

# Print MSE values
print('MSE Values:')
for i, mse in enumerate(mses):
    print(f'Architecture {i+1}: {mse}')
