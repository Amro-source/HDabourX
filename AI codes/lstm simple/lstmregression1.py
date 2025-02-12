from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

# Define the sample size and feature size
sampleSize = 100
featureSize = 10

# Create some sample data
np.random.seed(0)
X = np.random.rand(sampleSize, featureSize, 1)
y = np.random.rand(sampleSize, 1)

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(featureSize, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model with verbose output
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

