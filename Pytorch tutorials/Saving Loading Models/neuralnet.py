import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# Save model's state_dict
PATH = 'model.pt'
torch.save(model.state_dict(), PATH)

# Load model's state_dict
model_loaded = TheModelClass()  # Instantiate a new model
model_loaded.load_state_dict(torch.load(PATH))  # Load the state dict
model_loaded.eval()  # Set to evaluation mode

print("Model loaded successfully.")

# Export to TorchScript and save
model_scripted = torch.jit.script(model_loaded)  # Export to TorchScript
model_scripted.save('model_scripted.pt')  # Save

# Load the TorchScript model
model_jit = torch.jit.load('model_scripted.pt')  # Load the scripted model
model_jit.eval()  # Set to evaluation mode

print("TorchScript model loaded successfully.")