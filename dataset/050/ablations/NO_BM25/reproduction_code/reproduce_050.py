import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

tf.random.set_seed(42)

latent_dim = 100

generator = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(28 * 28 * 1, activation='tanh')
])

discriminator = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def call(self, inputs):
        return self.discriminator(self.generator(inputs))

gan = GAN(generator, discriminator, latent_dim)
gan.compile(optimizer='adam', loss='binary_crossentropy')

def dataloader():
    for _ in range(10):
        yield [tf.random.normal((28, 28, 1)) for _ in range(32)]

batch_size = 32
gan.fit(dataloader(), epochs=1)