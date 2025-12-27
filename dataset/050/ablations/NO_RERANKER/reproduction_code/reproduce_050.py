# Install dependencies
# pip install keras==3.4.1
# pip install tensorflow
# pip install keras-cv

import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from tqdm.auto import tqdm

tf.random.set_seed(42)

latent_dim = 100
batch_size = 32
epochs = 1
dataloader = [tf.random.normal((batch_size, 64, 64, 3))]

class SimpleGAN:
    def __init__(self):
        self.generator = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(latent_dim,)),
            keras.layers.Dense(64 * 64 * 3, activation='tanh'),
            keras.layers.Reshape((64, 64, 3))
        ])
        self.discriminator = keras.Sequential([
            keras.layers.Flatten(input_shape=(64, 64, 3)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def compile(self, optimizer, loss):
        self.discriminator.compile(optimizer=optimizer, loss=loss)

gan = SimpleGAN()
gan.compile(optimizer=keras.optimizers.Adam(), loss='binary_crossentropy')

gan.fit(dataloader, epochs=1)