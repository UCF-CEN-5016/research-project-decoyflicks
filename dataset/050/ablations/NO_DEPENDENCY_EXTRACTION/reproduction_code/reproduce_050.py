import os
import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import bounding_box
from keras_cv import visualization
import torch
import torchvision

tf.random.set_seed(42)
torch.manual_seed(42)

latent_dim = 100

def create_generator():
    model = keras.Sequential([
        keras.layers.Input(shape=(latent_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(28 * 28 * 1, activation='sigmoid'),
        keras.layers.Reshape((28, 28, 1))
    ])
    return model

def create_discriminator():
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    
    def compile(self):
        self.discriminator.trainable = False
        self.gan = keras.Sequential([self.generator, self.discriminator])
        self.gan.compile(optimizer='adam', loss='binary_crossentropy')

def dataloader():
    real_images = [tf.random.uniform((28, 28, 1)) for _ in range(10)]  # List of images
    return real_images

generator = create_generator()
discriminator = create_discriminator()
gan = GAN(generator, discriminator)
gan.compile()

real_images = dataloader()
gan.fit(real_images, epochs=1)  # This will raise the AttributeError