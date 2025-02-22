import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # Second convolutional layer

        # Dropout layers
        self.dropout1 = nn.Dropout2d(0.25)   # Dropout after pooling
        self.dropout2 = nn.Dropout2d(0.5)    # Dropout after first fully connected layer

        # Calculate the input size for the first fully connected layer dynamically
        # Assuming an input image size of (1, 28, 28)
        input_height, input_width = 28, 28  # Initial input dimensions
        conv1_output_height = (input_height - 3 + 2 * 0) // 1 + 1  # Conv1 output height
        conv1_output_width = (input_width - 3 + 2 * 0) // 1 + 1     # Conv1 output width

        conv2_output_height = (conv1_output_height - 3 + 2 * 0) // 1 + 1  # Conv2 output height
        conv2_output_width = (conv1_output_width - 3 + 2 * 0) // 1 + 1    # Conv2 output width

        pool_output_height = conv2_output_height // 2  # Max pooling reduces height by half
        pool_output_width = conv2_output_width // 2     # Max pooling reduces width by half

        fc1_input_size = 64 * pool_output_height * pool_output_width  # Flatten the output

        # Fully connected layers
        self.fc1 = nn.Linear(fc1_input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 10)              # Second fully connected layer

    def forward(self, x):
        # Pass data through conv1 and apply ReLU activation
        x = self.conv1(x)
        x = F.relu(x)

        # Pass data through conv2 and apply ReLU activation
        x = self.conv2(x)
        x = F.relu(x)

        # Apply max pooling and dropout1
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)

        # Pass data through fc1, apply ReLU activation, and dropout2
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Pass data through fc2 and apply log softmax
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Create an instance of the network
my_nn = Net()

# Print the network architecture
print(my_nn)

# Create random data to simulate a 28x28 grayscale image
random_data = torch.rand((1, 1, 28, 28))

# Pass the random data through the network
result = my_nn(random_data)

# Print the result
print(result)


optimizer = optim.SGD(my_nn.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in my_nn.state_dict():
    print(param_tensor, "\t", my_nn.state_dict()[param_tensor].size())

print()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])