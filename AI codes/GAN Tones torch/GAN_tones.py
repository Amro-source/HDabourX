

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as write_wav

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define device (CPU in this case)
device = torch.device("cpu")

# Hyperparameters (define variables for flexibility)
latent_dim = 10  # Dimension of the noise vector
signal_length = 44100  # Length of the generated signal (1 second at 44.1 kHz)
batch_size = 64
epochs = 10000
sample_interval = 1000
num_samples = 1000  # Number of samples for precomputed dataset

# Generate real data (tones)
def generate_real_tones(n_samples):
    """Generate real tones (sine waves) as ground truth."""
    frequencies = np.random.uniform(100, 1000, size=(n_samples, 1))  # Random frequencies
    t = np.linspace(0, 1, signal_length, endpoint=False).reshape(1, -1)  # Time vector
    X = np.sin(2 * np.pi * frequencies * t)  # Vectorized sine wave generation
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.ones((n_samples, 1), dtype=torch.float32).to(device)  # Real data label
    return X, y

# Generate fake data
def generate_fake_data(generator, latent_dim, n_samples):
    """Generate fake data using the generator."""
    noise = torch.randn(n_samples, latent_dim, dtype=torch.float32).to(device)
    X = generator(noise)
    y = torch.zeros((n_samples, 1), dtype=torch.float32).to(device)  # Fake data label
    return X, y

# Build the Generator (as a model)
generator = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, signal_length),
    nn.Tanh()  # Output in the range [-1, 1]
).to(device)

# Build the Discriminator (as a model)
discriminator = nn.Sequential(
    nn.Linear(signal_length, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()  # Output a probability
).to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Precompute a dataset of real tones
real_dataset = []
for _ in range(num_samples // batch_size):  # Generate enough batches
    X_real, _ = generate_real_tones(batch_size)
    real_dataset.append(X_real)

# Training loop
def train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval):
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch}")  # Debug print
        
        # Train Discriminator
        optimizer_D.zero_grad()
        
        # Sample real data from the precomputed dataset
        X_real = real_dataset[epoch % len(real_dataset)]
        y_real = torch.ones((batch_size, 1), dtype=torch.float32).to(device)
        real_output = discriminator(X_real)
        d_loss_real = criterion(real_output, y_real)
        print(f"Epoch {epoch}, D Loss Real: {d_loss_real.item():.4f}")  # Debug print
        
        # Generate fake data
        X_fake, y_fake = generate_fake_data(generator, latent_dim, batch_size)
        fake_output = discriminator(X_fake.detach())
        d_loss_fake = criterion(fake_output, y_fake)
        print(f"Epoch {epoch}, D Loss Fake: {d_loss_fake.item():.4f}")  # Debug print
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        if torch.isnan(d_loss) or torch.isinf(d_loss):
            print("Discriminator loss is NaN or Inf. Stopping training.")
            break
        d_loss.backward()
        optimizer_D.step()
        print(f"Epoch {epoch}, Updated Discriminator")  # Debug print
        
        # Train Generator
        optimizer_G.zero_grad()
        X_fake, y_gen = generate_fake_data(generator, latent_dim, batch_size)
        fake_output = discriminator(X_fake)
        g_loss = criterion(fake_output, y_gen)
        if torch.isnan(g_loss) or torch.isinf(g_loss):
            print("Generator loss is NaN or Inf. Stopping training.")
            break
        g_loss.backward()
        optimizer_G.step()
        print(f"Epoch {epoch}, Updated Generator")  # Debug print
        
        # Print progress
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    # Save models at the end of training
    torch.save(generator.state_dict(), 'generator_tones.pth')
    torch.save(discriminator.state_dict(), 'discriminator_tones.pth')
    print("Generator and Discriminator models saved.")
    
    # Generate and save a final tone
    noise = torch.randn(1, latent_dim, dtype=torch.float32).to(device)
    with torch.no_grad():
        generated_signal = generator(noise).cpu().numpy().flatten()
    
    # Normalize and save the audio
    normalized_signal = (generated_signal / np.max(np.abs(generated_signal))) * 32767
    normalized_signal = np.clip(normalized_signal, -32767, 32767).astype(np.int16)
    write_wav("generated_tone.wav", 44100, normalized_signal)
    print("Generated tone saved as 'generated_tone.wav'.")

# Main execution
if __name__ == "__main__":
    train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval)