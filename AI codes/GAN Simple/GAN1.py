import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
latent_dim = 10
batch_size = 64
epochs = 10000
sample_interval = 1000

# Generate real data
def generate_real_data(n_samples):
    X = np.random.normal(loc=0, scale=1, size=(n_samples, 1)).astype(np.float32)
    y = np.ones((n_samples, 1), dtype=np.float32)
    return X, y

# Generate fake data
def generate_fake_data(generator, latent_dim, n_samples):
    noise = np.random.normal(0, 1, (n_samples, latent_dim)).astype(np.float32)
    X = generator(noise, training=True)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    return X.numpy(), y

# Build the Generator
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(16),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dense(1, activation='linear')
    ])
    return model

# Build the Discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(16),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile Models
discriminator = build_discriminator()
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
    metrics=['accuracy']
)
discriminator.optimizer = discriminator.optimizer  # Explicitly store the optimizer

generator = build_generator(latent_dim)

# Freeze discriminator for GAN training
discriminator.trainable = False

# Build & Compile GAN
gan = models.Sequential([generator, discriminator])
gan.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
)
generator.optimizer = gan.optimizer  # Use the same optimizer for the generator

# Custom Training Loop with Gradient Watch Fix
def train_gan(generator, discriminator, gan, latent_dim, epochs, batch_size, sample_interval):
    for epoch in range(epochs):
        # Train Discriminator
        X_real, y_real = generate_real_data(batch_size // 2)
        X_fake, y_fake = generate_fake_data(generator, latent_dim, batch_size // 2)
        X_combined = np.vstack((X_real, X_fake)).astype(np.float32)
        y_combined = np.vstack((y_real, y_fake)).astype(np.float32)

        with tf.GradientTape() as tape:
            y_pred = discriminator(X_combined, training=True)
            d_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_combined, y_pred))

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        if grads is not None and len(grads) > 0:  # Ensure gradients exist
            discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        else:
            print("Warning: No gradients computed for discriminator!")

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        y_gen = np.ones((batch_size, 1), dtype=np.float32)

        with tf.GradientTape() as tape:
            fake_data = generator(noise, training=True)
            y_pred_fake = discriminator(fake_data, training=False)
            g_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_gen, y_pred_fake))

        grads = tape.gradient(g_loss, generator.trainable_variables)
        if grads is not None and len(grads) > 0:  # Ensure gradients exist
            generator.optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        else:
            print("Warning: No gradients computed for generator!")

        # Print progress
        if epoch % sample_interval == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
            sample_and_plot(generator, latent_dim)

# Visualize generated data
def sample_and_plot(generator, latent_dim):
    noise = np.random.normal(0, 1, (100, latent_dim)).astype(np.float32)
    generated_data = generator(noise, training=False).numpy()
    plt.figure(figsize=(8, 4))
    plt.hist(generated_data, bins=20, alpha=0.5, label='Generated Data')
    plt.hist(np.random.normal(loc=0, scale=1, size=100), bins=20, alpha=0.5, label='Real Data')
    plt.legend()
    plt.title('Generated vs Real Data Distribution')
    plt.show()

# Main execution
if __name__ == "__main__":
    train_gan(generator, discriminator, gan, latent_dim, epochs, batch_size, sample_interval)