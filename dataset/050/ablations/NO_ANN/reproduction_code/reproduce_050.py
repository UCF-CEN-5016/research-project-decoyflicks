import numpy as np
import keras
from keras import layers
import tensorflow as tf

np.random.seed(42)

latent_dim = 100

class SimpleGAN:
    def __init__(self):
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.compile_models()

    def build_discriminator(self):
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=latent_dim),
            layers.Dense(28 * 28 * 1, activation='tanh'),
            layers.Reshape((28, 28, 1))
        ])
        return model

    def compile_models(self):
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.generator.compile(loss='binary_crossentropy', optimizer='adam')

    def fit(self, dataloader, epochs):
        for epoch in range(epochs):
            for real_images in dataloader:
                # This line will raise the AttributeError
                print(real_images.shape)

dataloader = [np.random.rand(32, 28, 28, 1) for _ in range(10)]
gan = SimpleGAN()
gan.fit(dataloader, epochs=1)