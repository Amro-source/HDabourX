import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io.wavfile import write as write_wav
import psutil
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define device (CPU in this case)
device = torch.device("cpu")

# Hyperparameters
latent_dim = 10
signal_length = 44100
batch_size = 64
epochs = 1000
sample_interval = 1000
num_samples = 1000

# Generate real data (tones)
def generate_real_tones(n_samples):
    frequencies = np.random.uniform(100, 1000, size=(n_samples, 1))
    t = np.linspace(0, 1, signal_length, endpoint=False).reshape(1, -1)
    X = np.sin(2 * np.pi * frequencies * t)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.ones((n_samples, 1), dtype=torch.float32).to(device)
    return X, y

# Generate fake data
def generate_fake_data(generator, latent_dim, n_samples):
    noise = torch.randn(n_samples, latent_dim, dtype=torch.float32).to(device)
    X = generator(noise)
    y = torch.zeros((n_samples, 1), dtype=torch.float32).to(device)
    return X, y

# Build the Generator (more complex architecture)
generator = nn.Sequential(
    nn.Linear(latent_dim, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 1024),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, signal_length),
    nn.Tanh()
).to(device)

# Build the Discriminator (more complex architecture, no Sigmoid)
discriminator = nn.Sequential(
    nn.Linear(signal_length, 1024),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 1)
).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

# Loss function (BCEWithLogitsLoss for logits)
criterion = nn.BCEWithLogitsLoss()

# Precompute a dataset of real tones
real_dataset = []
for _ in range(num_samples // batch_size):
    X_real, _ = generate_real_tones(batch_size)
    real_dataset.append(X_real)

# Function to monitor memory usage
def print_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 ** 2):.2f} MB")

# Function to save models
def save_models(generator, discriminator, epoch):
    try:
        # Ensure the directory exists
        os.makedirs("models", exist_ok=True)
        
        # Save generator and discriminator
        generator_path = f"models/generator_epoch_{epoch}.pth"
        discriminator_path = f"models/discriminator_epoch_{epoch}.pth"
        torch.save(generator.state_dict(), generator_path)
        torch.save(discriminator.state_dict(), discriminator_path)
        print(f"Models saved: {generator_path}, {discriminator_path}")
    except Exception as e:
        print(f"Error saving models: {e}")

# Training loop
def train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval):
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch}")  # Debug print at the start of each epoch
        
        # Train Discriminator (update every other epoch)
        if epoch % 2 == 0:
            optimizer_D.zero_grad()
            
            # Real data
            X_real = real_dataset[epoch % len(real_dataset)]
            if X_real.shape[0] != batch_size:
                print(f"Invalid batch size for X_real: {X_real.shape}. Expected: {batch_size}")
                exit()
            y_real = torch.ones((batch_size, 1), dtype=torch.float32).to(device)
            real_output = discriminator(X_real)
            d_loss_real = criterion(real_output, y_real)
            
            # Fake data
            X_fake, y_fake = generate_fake_data(generator, latent_dim, batch_size)
            fake_output = discriminator(X_fake.detach())
            d_loss_fake = criterion(fake_output, y_fake)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            if torch.isnan(d_loss) or torch.isinf(d_loss):
                print("Discriminator loss is NaN or Inf. Stopping training.")
                break
            
            # Backward pass and optimizer step
            d_loss.backward()
            optimizer_D.step()
            
            # Print discriminator loss
            print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}")
        
        # Train Generator
        optimizer_G.zero_grad()
        X_fake, y_gen = generate_fake_data(generator, latent_dim, batch_size)
        fake_output = discriminator(X_fake)
        g_loss = criterion(fake_output, y_gen)
        if torch.isnan(g_loss) or torch.isinf(g_loss):
            print("Generator loss is NaN or Inf. Stopping training.")
            break
        
        # Backward pass and optimizer step
        g_loss.backward()
        optimizer_G.step()
        
        # Print generator loss
        print(f"Epoch {epoch}, G Loss: {g_loss.item():.4f}")
        
        # Save intermediate models
        if epoch % sample_interval == 0:
            save_models(generator, discriminator, epoch)
        
        print(f"Completed Epoch {epoch}")  # Debug print at the end of each epoch
        
        # Monitor memory usage
        print_memory_usage()

    # Save final models
    save_models(generator, discriminator, "final")

# Main execution
if __name__ == "__main__":
    train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval)