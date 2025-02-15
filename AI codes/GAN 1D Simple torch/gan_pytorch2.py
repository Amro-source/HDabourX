import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
latent_dim = 10
batch_size = 64
epochs = 10000
sample_interval = 1000

# Generate real data
def generate_real_data(n_samples):
    X = np.random.normal(loc=0, scale=1, size=(n_samples, 1)).astype(np.float32)
    y = np.ones((n_samples, 1), dtype=np.float32)
    return torch.tensor(X), torch.tensor(y)

# Generate fake data
def generate_fake_data(generator, latent_dim, n_samples):
    noise = torch.randn(n_samples, latent_dim, dtype=torch.float32)
    X = generator(noise)
    y = torch.zeros((n_samples, 1), dtype=torch.float32)
    return X, y

# Build the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# Build the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(latent_dim)
discriminator = Discriminator()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
def train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval):
    for epoch in range(epochs):
        # Train Discriminator
        optimizer_D.zero_grad()

        # Real data
        X_real, y_real = generate_real_data(batch_size // 2)
        real_output = discriminator(X_real)
        d_loss_real = criterion(real_output, y_real)

        # Fake data
        X_fake, y_fake = generate_fake_data(generator, latent_dim, batch_size // 2)
        fake_output = discriminator(X_fake.detach())
        d_loss_fake = criterion(fake_output, y_fake)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        X_fake, y_gen = generate_fake_data(generator, latent_dim, batch_size)
        fake_output = discriminator(X_fake)
        g_loss = criterion(fake_output, y_gen)
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            sample_and_plot(generator, latent_dim)

    # Save models at the end of training
    torch.save(generator.state_dict(), 'generator_final.pth')
    torch.save(discriminator.state_dict(), 'discriminator_final.pth')
    print("Generator and Discriminator models saved.")

# Visualize generated data (non-blocking)
def sample_and_plot(generator, latent_dim):
    noise = torch.randn(100, latent_dim, dtype=torch.float32)
    generated_data = generator(noise).detach().numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(generated_data, bins=20, alpha=0.5, label='Generated Data')
    plt.hist(np.random.normal(loc=0, scale=1, size=100), bins=20, alpha=0.5, label='Real Data')
    plt.legend()
    plt.title('Generated vs Real Data Distribution')
    plt.show(block=False)  # Show the plot without blocking
    plt.pause(3)  # Pause briefly to update the plot
    plt.close()  # Close the plot to avoid blocking

# Main execution
if __name__ == "__main__":
    train_gan(generator, discriminator, optimizer_G, optimizer_D, latent_dim, epochs, batch_size, sample_interval)