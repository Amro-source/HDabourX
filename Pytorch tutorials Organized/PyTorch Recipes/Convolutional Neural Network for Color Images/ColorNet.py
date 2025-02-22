import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorImageNet(nn.Module):
    def __init__(self):
        super(ColorImageNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # Input: 3 channels, Output: 32 feature maps, Kernel: 3x3, Stride: 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Input: 32 channels, Output: 64 feature maps, Kernel: 3x3, Stride: 1
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)  # Use nn.Dropout for fully connected layers
        self.dropout2 = nn.Dropout(0.5)   # Use nn.Dropout for fully connected layers
        # Fully connected layers
        fc1_input_size = 64 * 14 * 14  # Calculated based on the output of the convolutional layers
        self.fc1 = nn.Linear(fc1_input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 10)              # Second fully connected layer (10 classes)

    def forward(self, x):
        # Pass data through conv1 and apply ReLU activation
        x = self.conv1(x)
        x = F.relu(x)
        # Pass data through conv2 and apply ReLU activation
        x = self.conv2(x)
        x = F.relu(x)
        # Apply max pooling and dropout1
        x = F.max_pool2d(x, 2)  # Reduce spatial dimensions by half
        x = self.dropout1(x)    # Use nn.Dropout here
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        # Pass data through fc1, apply ReLU activation, and dropout2
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)    # Use nn.Dropout here
        # Pass data through fc2 and apply log softmax
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Create an instance of the network
my_color_nn = ColorImageNet()

# Print the network architecture
print(my_color_nn)

# Create random data to simulate a 32x32 color image
random_data = torch.rand((1, 3, 32, 32))  # Batch size: 1, Channels: 3, Height: 32, Width: 32

# Pass the random data through the network
result = my_color_nn(random_data)

# Print the result
print(result)