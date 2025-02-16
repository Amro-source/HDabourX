import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io.wavfile import read as read_wav

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
latent_dim = 10
signal_length = 44100  # Assuming all audio files are 1 second long at 44.1 kHz
batch_size = 64
epochs = 50000
sample_interval = 1000

# Load real data (tones) from .wav files
def load_real_tones_from_files(data_dir, signal_length=44100):
    """Load real tones from .wav files in the specified directory."""
    X = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(data_dir, file_name)
            sample_rate, signal = read_wav(file_path)
            if len(signal) != signal_length:
                print(f"Skipping {file_name}: Incorrect length ({len(signal)} samples).")
                continue
            X.append(signal.astype(np.float32) / 32767.0)  # Normalize to [-1, 1]
    return np.array(X)

# Build the Generator
generator = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, signal_length),
    nn.Tanh()
)

# Build the Discriminator
discriminator = nn.Sequential(
    nn.Linear(signal_length, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1)
)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005, betas=(0.5, 0.999))

# Loss function (MSE for smoother convergence)
criterion = nn.MSELoss()

# Compute gradient penalty for WGAN-GP
def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Function to compute gradient norms
def compute_gradient_norms(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

# Training loop
def train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval, data_dir):
    device = torch.device("cpu")
    generator.to(device)
    discriminator.to(device)

    # Load real tones from .wav files
    real_tones = load_real_tones_from_files(data_dir, signal_length=signal_length)
    if len(real_tones) == 0:
        raise ValueError(f"No valid audio files found in {data_dir}. Ensure files are 1-second long at 44.1 kHz.")
    
    # Convert to tensor and split into batches
    real_tones = torch.tensor(real_tones, dtype=torch.float32)
    num_samples = len(real_tones)
    print(f"Loaded {num_samples} real tones for training.")

    # Precompute dataset indices for shuffling
    dataset_indices = np.arange(num_samples)

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch}")  # Debug print

        # Shuffle dataset indices
        np.random.shuffle(dataset_indices)

        for i in range(0, num_samples, batch_size):
            batch_indices = dataset_indices[i:i + batch_size]
            batch_size_current = len(batch_indices)  # Handle last incomplete batch

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real data
            X_real = real_tones[batch_indices].to(device)
            y_real = torch.full((batch_size_current, 1), 0.9, dtype=torch.float32).to(device)  # Smoothed real labels
            real_output = discriminator(X_real)
            d_loss_real = criterion(real_output, y_real)

            # Fake data
            noise = torch.randn(batch_size_current, latent_dim, dtype=torch.float32).to(device)
            X_fake = generator(noise).detach()
            y_fake = torch.full((batch_size_current, 1), 0.1, dtype=torch.float32).to(device)  # Smoothed fake labels
            fake_output = discriminator(X_fake)
            d_loss_fake = criterion(fake_output, y_fake)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, X_real, X_fake, device)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake + 10 * gradient_penalty

            # Debug: Check for NaN/Inf losses
            if torch.isnan(d_loss).any() or torch.isinf(d_loss).any():
                print(f"Epoch {epoch}, Discriminator Loss is NaN or Inf: {d_loss}")
                return

            d_loss.backward()
            optimizer_D.step()

            # Debug: Print discriminator loss and gradient norm
            d_grad_norm = compute_gradient_norms(discriminator)
            print(f"Epoch {epoch}, Batch {i // batch_size}, D Loss: {d_loss.item():.4f}, D Grad Norm: {d_grad_norm:.4f}")

            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size_current, latent_dim, dtype=torch.float32).to(device)
            X_fake = generator(noise)
            fake_output = discriminator(X_fake)
            g_loss = criterion(fake_output, torch.full((batch_size_current, 1), 0.9, dtype=torch.float32).to(device))  # Smoothed target

            # Debug: Check for NaN/Inf losses
            if torch.isnan(g_loss).any() or torch.isinf(g_loss).any():
                print(f"Epoch {epoch}, Generator Loss is NaN or Inf: {g_loss}")
                return

            g_loss.backward()
            optimizer_G.step()

            # Debug: Print generator loss and gradient norm
            g_grad_norm = compute_gradient_norms(generator)
            print(f"Epoch {epoch}, Batch {i // batch_size}, G Loss: {g_loss.item():.4f}, G Grad Norm: {g_grad_norm:.4f}")

        # Print progress
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    torch.save(generator.state_dict(), 'models/generator_final.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator_final.pth')

if __name__ == "__main__":
    # Directory containing the generated audio files
    data_dir = "generated_tones"

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Generate audio files first using generate_data.py.")

    # Start training
    train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval, data_dir)