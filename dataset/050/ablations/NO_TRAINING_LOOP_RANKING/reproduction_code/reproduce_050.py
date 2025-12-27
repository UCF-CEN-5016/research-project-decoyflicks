import numpy as np
import keras
from keras import layers

latent_dim = 100

generator = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
    layers.Dense(28 * 28 * 1, activation='sigmoid'),
    layers.Reshape((28, 28, 1))
])

discriminator = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(1, activation='sigmoid')
])

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss='binary_crossentropy')
    
    def compile(self):
        self.generator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss='binary_crossentropy')

    def train_step(self, real_images):
        noise = np.random.normal(0, 1, (len(real_images), latent_dim))
        generated_images = self.generator(noise)
        real_labels = np.ones((len(real_images), 1))
        fake_labels = np.zeros((len(real_images), 1))
        d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = self.discriminator.train_on_batch(generated_images, fake_labels)
        return d_loss_real, d_loss_fake

gan = GAN(generator, discriminator)
gan.compile()

batch_size = 32
dataloader = [np.random.rand(batch_size, 28, 28, 1)]  # List of images

gan.train_step(dataloader)

gan.fit(dataloader, epochs=1)  # This line will raise the AttributeError