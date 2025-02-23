import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Hyper-parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # First convolutional layer: 1 input channel, 16 output channels, kernel size 5x5
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        # Max pooling layer with kernel size 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layer: input size = 16 * 14 * 14 (after pooling), output size = 10 classes
        self.fc = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x):
        # Apply first convolutional layer, ReLU activation, and max pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 16 * 14 * 14)
        # Apply fully connected layer
        x = self.fc(x)
        return x

# Initialize the CNN model
model = CNNModel()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f} %')

# Save the model checkpoint
torch.save(model.state_dict(), 'cnn_model.ckpt')