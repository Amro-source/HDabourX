LSTM Time Series Prediction Model

Overview
This repository contains a simple implementation of a Long Short-Term Memory (LSTM) neural network in PyTorch for time series prediction. The model is trained on a synthetic dataset and saved as a TorchScript model for deployment.

Requirements
- Python 3.6+
- PyTorch 1.9+
- NumPy
- Matplotlib

Usage
1. Clone the repository: git clone https://github.com/your-username/lstm-time-series-prediction.git
2. Install the required packages: pip install -r requirements.txt
3. Run the training script: python lstmprediction.py
4. Evaluate the model: python evaluate_model.py

Model Architecture
The LSTM model consists of the following layers:

- An LSTM layer with 50 hidden units and a batch size of 1
- A fully connected layer with 1 output unit

Training
The model is trained on a synthetic dataset generated using the generate_input_data function. The training process involves the following steps:

- Initialize the model, optimizer, and loss function
- Train the model for 100 epochs using the Adam optimizer and mean squared error loss function
- Save the trained model as a TorchScript model

Deployment
The trained model can be deployed using the torch.jit.load function. The evaluate_model.py script demonstrates how to load the saved model and use it to make predictions on new data.

Contributing
Contributions are welcome! If you'd like to contribute to this repository, please fork the repository and submit a pull request.

License
This repository is licensed under the MIT License.