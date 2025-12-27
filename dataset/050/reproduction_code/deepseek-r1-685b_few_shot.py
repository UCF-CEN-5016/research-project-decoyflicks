import keras
from keras import layers
import numpy as np

# Minimal GAN implementation reproducing the issue
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def train_step(self, real_images):
        # Bug occurs here - real_images comes as a list
        batch_size = real_images.shape[0]  # Fails because real_images is a list
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim)
        )
        return {}

# Create minimal models
latent_dim = 32
generator = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu")
])
discriminator = keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(1)
])

# Create GAN and compile
gan = GAN(discriminator, generator, latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(),
    g_optimizer=keras.optimizers.Adam()
)

# Create dummy data that matches the bug condition
dataloader = [[np.random.rand(32, 128)]]  # Wrapped in list to reproduce issue

# This will trigger the AttributeError
gan.fit(dataloader, epochs=1)