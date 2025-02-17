import os
import copy
import torch
import torch.nn as nn
import numpy as np
from scipy.io.wavfile import read as read_wav
import matplotlib.pyplot as plt
import time

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
checkpoint_interval = 10  # Save checkpoint every 10 epochs

# Ensure directory exists
def ensure_directory_exists(directory):
    """
    Ensures that the specified directory exists.
    Retries up to 5 times if creation fails.
    """
    retries = 5
    while retries > 0:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Successfully created or verified directory: {directory}")
            return True
        except FileExistsError:
            print(f"Directory already exists: {directory}")
            return True
        except Exception as e:
            print(f"Failed to create directory {directory}: {e}. Retrying...")
            retries -= 1
            time.sleep(1)
    raise RuntimeError(f"Could not create directory {directory} after multiple attempts.")

# Define the checkpoints directory
CHECKPOINT_DIR = os.path.abspath("checkpoints")
ensure_directory_exists(CHECKPOINT_DIR)

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
            X.append(signal.astype(np.float32))  # Normalize to [-1, 1]
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
            nn.Linear(256, 1),
            nn.Sigmoid()  # Use sigmoid for binary classification
        )

    def forward(self, x):
        return self.model(x)

# Save checkpoint
def save_checkpoint(generator, discriminator, population, fitness, epoch, checkpoint_dir=CHECKPOINT_DIR):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'population': population,
        'fitness': fitness,
        'epoch': epoch
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.")

# Load checkpoint
def load_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    # Debug: Check if the directory exists
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' does not exist. Starting training from scratch.")
        return None, None, None, None, 0

    try:
        # List checkpoint files in the directory
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir)
            if os.path.isfile(os.path.join(checkpoint_dir, f)) and f.startswith("checkpoint_epoch_")
        ]
        if not checkpoint_files:
            print("No checkpoints found. Starting training from scratch.")
            return None, None, None, None, 0

        # Get the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        return (
            checkpoint.get('generator_state_dict'),
            checkpoint.get('discriminator_state_dict'),
            checkpoint.get('population'),
            checkpoint.get('fitness'),
            checkpoint.get('epoch', 0)
        )
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return None, None, None, None, 0

# ABC Algorithm
def abc_algorithm(generator, discriminator, real_data, latent_dim, n_employed, n_onlookers, n_scouts, limit, epochs):
    device = torch.device("cpu")
    generator.to(device)
    discriminator.to(device)

    # Initialize population (state dictionaries of the generator)
    population = [copy.deepcopy(generator.state_dict()) for _ in range(n_employed)]
    fitness = [evaluate_fitness(generator, discriminator, real_data, latent_dim, weights) for weights in population]
    best_solution = population[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    # Resume from checkpoint if available
    gen_state_dict, disc_state_dict, loaded_population, loaded_fitness, start_epoch = load_checkpoint()

    if gen_state_dict is not None:
        generator.load_state_dict(gen_state_dict)
        discriminator.load_state_dict(disc_state_dict)
        population = loaded_population
        fitness = loaded_fitness

    for epoch in range(start_epoch, epochs):
        # Employed bees phase
        for i in range(n_employed):
            candidate = perturb_state_dict(population[i])
            candidate_fitness = evaluate_fitness(generator, discriminator, real_data, latent_dim, candidate)
            if candidate_fitness > fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness

        # Onlooker bees phase
        probabilities = np.array(fitness) / np.sum(fitness)
        for i in range(n_onlookers):
            idx = np.random.choice(range(n_employed), p=probabilities)
            candidate = perturb_state_dict(population[idx])
            candidate_fitness = evaluate_fitness(generator, discriminator, real_data, latent_dim, candidate)
            if candidate_fitness > fitness[idx]:
                population[idx] = candidate
                fitness[idx] = candidate_fitness

        # Scout bees phase
        for i in range(n_scouts):
            idx = np.argmin(fitness)
            if np.random.rand() < 0.1:  # Abandon solution
                population[idx] = copy.deepcopy(generator.state_dict())
                fitness[idx] = evaluate_fitness(generator, discriminator, real_data, latent_dim, population[idx])

        # Update best solution
        if np.max(fitness) > best_fitness:
            best_solution = population[np.argmax(fitness)]
            best_fitness = np.max(fitness)

        print(f"Epoch {epoch + 1}/{epochs}, Best Fitness: {best_fitness}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(generator, discriminator, population, fitness, epoch + 1)

    return best_solution

# Perturb state dictionary (add noise to weights)
def perturb_state_dict(state_dict):
    perturbed_state_dict = {}
    for key, value in state_dict.items():
        perturbed_state_dict[key] = value + torch.randn_like(value) * 0.1
    return perturbed_state_dict

# Fitness function based on generator and discriminator losses
def evaluate_fitness(generator, discriminator, real_data, latent_dim, weights):
    device = torch.device("cpu")
    generator.load_state_dict(weights)
    generator.eval()

    # Generate fake data
    noise = torch.randn(real_data.size(0), latent_dim, dtype=torch.float32).to(device)
    fake_data = generator(noise)

    # Discriminator output
    real_output = discriminator(real_data).view(-1)
    fake_output = discriminator(fake_data).view(-1)

    # Adversarial loss (generator loss)
    generator_loss = -torch.mean(torch.log(fake_output + 1e-8))

    # Discriminator loss
    discriminator_loss = -torch.mean(torch.log(real_output + 1e-8) + torch.log(1 - fake_output + 1e-8))

    # Total fitness (maximize generator performance while minimizing discriminator loss)
    fitness = -(generator_loss.item() + 0.1 * discriminator_loss.item())
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

    # Train using ABC
    best_weights = abc_algorithm(generator, discriminator, real_tones, latent_dim, n_employed, n_onlookers, n_scouts, limit, epochs)

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