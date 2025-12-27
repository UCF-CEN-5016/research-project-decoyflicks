import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers

# Data loading and preprocessing
(x_train, _), (x_test, _) = datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape((x_train.shape[0], -1))  # flatten to (N, 784)

# Batch and latent configuration
GLOBAL_BATCH_SIZE = x_train.shape[0]
LATENT_DIM = 100

# Pre-generate some latent vectors (kept for parity with original)
random_latent_vectors = np.random.normal(size=(GLOBAL_BATCH_SIZE, LATENT_DIM))

# Discriminator model
discriminator = models.Sequential(
    [
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)

# Generator model
generator = models.Sequential(
    [
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(784, activation="sigmoid"),
    ],
    name="generator",
)

# Compile discriminator and generator (keeps original intent)
disc_optimizer = optimizers.Adam(learning_rate=0.0003)
gen_optimizer = optimizers.Adam(learning_rate=0.0003)
discriminator.compile(optimizer=disc_optimizer, loss="binary_crossentropy")
generator.compile(optimizer=gen_optimizer, loss="mean_squared_error")


class GAN(models.Model):
    def __init__(self, discriminator_model, generator_model, latent_dim):
        super(GAN, self).__init__(name="gan_model")
        self._discriminator = discriminator_model
        self._generator = generator_model
        self.latent_dim = latent_dim
        # placeholders for optional compile-time attributes
        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    @property
    def discriminator(self):
        return self._discriminator

    @discriminator.setter
    def discriminator(self, value):
        self._discriminator = value

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, value):
        self._generator = value

    def compile(self, d_optimizer=None, g_optimizer=None, loss_fn=None, **kwargs):
        # Store provided optimizers and loss function for bookkeeping (no-op for train_step)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        super(GAN, self).compile(**kwargs)

    def train_step(self, data):
        # Expect data to be a batch of real images already flattened
        real_images = data
        batch_size = tf.shape(real_images)[0]

        # Generate fake images from random latent vectors
        latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_images = self.generator(latent_vectors, training=False)

        # Train discriminator on real and fake images
        self.discriminator.trainable = True
        real_labels = np.ones((int(batch_size), 1), dtype="float32")
        fake_labels = np.zeros((int(batch_size), 1), dtype="float32")

        d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = self.discriminator.train_on_batch(fake_images, fake_labels)

        # Average discriminator loss
        try:
            d_loss_value = 0.5 * (d_loss_real + d_loss_fake)
        except Exception:
            # If loss returns a list (e.g., metrics), try to extract scalar
            d_loss_value = 0.5 * (np.array(d_loss_real).mean() + np.array(d_loss_fake).mean())

        # Train generator via the discriminator by labeling fakes as real
        self.discriminator.trainable = False
        trick_labels = np.ones((int(batch_size), 1), dtype="float32")
        g_loss = self.discriminator.train_on_batch(fake_images, trick_labels)

        return {"d_loss": d_loss_value, "g_loss": g_loss}


# Instantiate and compile the GAN wrapper
gan = GAN(discriminator_model=discriminator, generator_model=generator, latent_dim=LATENT_DIM)
gan.compile(d_optimizer=optimizers.Adam(), g_optimizer=optimizers.Adam(), loss_fn="binary_crossentropy")

# Train for 1 epoch using the full-batch size as in the original code
gan.fit(x_train, epochs=1, batch_size=GLOBAL_BATCH_SIZE)