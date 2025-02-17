import os
import torch
import torch.nn as nn
import numpy as np
from scipy.io.wavfile import read as read_wav
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
latent_dim = 10
signal_length = 44100  # Assuming all audio files are 1 second long at 44.1 kHz
batch_size = 128  # Increased batch size
epochs = 100  # Number of ABC iterations
n_employed = 10  # Number of employed bees
n_onlookers = 10  # Number of onlooker bees
n_scouts = 2  # Number of scout bees
limit = 10  # Limit for abandoning a solution

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
            # Normalize to [-1, 1]
            signal = signal.astype(np.float32) / np.iinfo(signal.dtype).max
            X.append(signal)
    return np.array(X)

# Build the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, signal_length),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Build the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(signal_length, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

# ABC Algorithm
def abc_algorithm(generator, discriminator, real_data, latent_dim, n_employed, n_onlookers, n_scouts, limit, epochs):
    device = torch.device("cpu")
    generator.to(device)
    discriminator.to(device)

    # Initialize population (weights of the generator and discriminator)
    population = [list(generator.parameters()) for _ in range(n_employed)]
    fitness = [evaluate_fitness(generator, discriminator, real_data, latent_dim, params) for params in population]

    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    for epoch in range(epochs):
        # Employed bees phase
        for i in range(n_employed):
            candidate = [param + torch.randn_like(param) * 0.1 for param in population[i]]
            candidate_fitness = evaluate_fitness(generator, discriminator, real_data, latent_dim, candidate)
            if candidate_fitness > fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness

        # Onlooker bees phase
        probabilities = np.array(fitness) / np.sum(fitness)
        for i in range(n_onlookers):
            idx = np.random.choice(range(n_employed), p=probabilities)
            candidate = [param + torch.randn_like(param) * 0.1 for param in population[idx]]
            candidate_fitness = evaluate_fitness(generator, discriminator, real_data, latent_dim, candidate)
            if candidate_fitness > fitness[idx]:
                population[idx] = candidate
                fitness[idx] = candidate_fitness

        # Scout bees phase
        for i in range(n_scouts):
            idx = np.argmin(fitness)
            if np.random.rand() < 0.1:  # Abandon solution
                population[idx] = [torch.randn_like(param) for param in population[idx]]
                fitness[idx] = evaluate_fitness(generator, discriminator, real_data, latent_dim, population[idx])

        # Update best solution
        if np.max(fitness) > best_fitness:
            best_solution = population[np.argmax(fitness)]
            best_fitness = np.max(fitness)

        print(f"Epoch {epoch + 1}/{epochs}, Best Fitness: {best_fitness}")

    return best_solution

# Fitness function for ABC
def evaluate_fitness(generator, discriminator, real_data, latent_dim, weights):
    device = torch.device("cpu")
    for i, param in enumerate(generator.parameters()):
        param.data = weights[i].data.clone()

    generator.eval()

    # Generate fake data with the same batch size as real_data
    noise = torch.randn(real_data.size(0), latent_dim, dtype=torch.float32).to(device)
    fake_data = generator(noise)

    # Discriminator output
    real_output = discriminator(real_data).view(-1)  # Flatten to match shape
    fake_output = discriminator(fake_data).view(-1)  # Flatten to match shape

    # Adversarial loss
    adv_loss = torch.mean(torch.log(real_output.clamp(min=1e-8)) + torch.log((1 - fake_output).clamp(min=1e-8)))

    # Spectral loss
    real_fft = torch.fft.rfft(real_data, norm='ortho')
    fake_fft = torch.fft.rfft(fake_data, norm='ortho')
    spectral_l = torch.mean(torch.abs(real_fft - fake_fft))

    # Total fitness (maximize)
    fitness = - (adv_loss + 0.1 * spectral_l).item()
    return fitness

# Training loop
def train_gan_with_abc(generator, discriminator, latent_dim, epochs, batch_size, data_dir):
    device = torch.device("cpu")
    generator.to(device)
    discriminator.to(device)

    # Load real tones from .wav files
    real_tones = load_real_tones_from_files(data_dir, signal_length=signal_length)
    if len(real_tones) == 0:
        raise ValueError(f"No valid audio files found in {data_dir}. Ensure files are 1-second long at 44.1 kHz.")

    # Convert to tensor
    real_tones = torch.tensor(real_tones, dtype=torch.float32).to(device)

    # Split real tones into batches
    real_batches = [real_tones[i:i + batch_size] for i in range(0, len(real_tones), batch_size)]

    # Train using ABC
    best_weights = None
    for epoch in range(epochs):
        for real_batch in real_batches:
            best_weights = abc_algorithm(
                generator, discriminator, real_batch, latent_dim,
                n_employed, n_onlookers, n_scouts, limit, 1  # Run ABC for 1 epoch per batch
            )

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

    # Initialize models
    generator = Generator(latent_dim)
    discriminator = Discriminator()

    # Start training
    train_gan_with_abc(generator, discriminator, latent_dim, epochs, batch_size, data_dir)