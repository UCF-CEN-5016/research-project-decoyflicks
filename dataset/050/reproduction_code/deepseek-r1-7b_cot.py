import os
import numpy as np
from tensorflow.keras import layers, models, datasets, utils

# Minimal reproduction code
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
batch_size = x_train.shape[0] if hasattr(x_train, 'shape') else len(x_train)
real_images = x_train[:batch_size]
random_latent_vectors = np.random.normal(
    size=(batch_size, 100))
discriminator = models.Sequential([
    layers.Dense(256, activation='relu')(layers.Input(shape=(784,))),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
generator = models.Sequential([
    layers.Dense(256, activation='relu')(layers.Input(shape=(100,))),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(784, activation='sigmoid')
])
discriminator.compile(
    optimizer(layers.Adam, learning_rate=0.0003),
    loss='binary_crossentropy'
)
generator.compile(
    optimizer(layers.Adam, learning_rate=0.0003),
    loss='mean_squared_error'
)

class GAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

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

    def train_step(self, data):
        real_images = data[0]
        fake_images = self.generator.predict(keras.random.normal(
            shape=(batch_size, self.latent_dim), 
            seed=self.seed_generator))
        
        # Train the discriminator
        discriminator.trainable = True
        d_loss_real = self.discriminator.train_on_batch(real_images, np.ones(batch_size))
        d_loss_fake = self.discriminator.train_on_batch(fake_images, np.zeros(batch_size))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        
        # Train the generator
        discriminator.trainable = False
        g_loss = self.discriminator.train_on_batch(fake_images, np.ones(2*batch_size))
        
        return {'d_loss': d_loss, 'g_loss': g_loss}

# Compile and train the GAN
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=100)
gan.compile(
    d_optimizer=layers.Adam,
    g_optimizer=layers.Adam,
    loss_fn='binary_crossentropy'
)
gan.fit(x_train, epochs=1, batch_size=batch_size)