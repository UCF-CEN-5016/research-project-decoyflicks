import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Minimal GAN implementation reproducing the issue
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        return {}

# Create minimal models
latent_dim = 32
generator = tf.keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(128, activation="relu")
])
discriminator = tf.keras.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(1)
])

# Create GAN and compile
gan = GAN(discriminator, generator, latent_dim)
gan.compile(
    optimizer= tf.keras.optimizers.Adam(),
    loss = 'binary_crossentropy'
)

# Create dummy data that matches the bug condition
dataloader = np.random.rand(32, 128)

# This will trigger the AttributeError
gan.fit(dataloader, epochs=1)