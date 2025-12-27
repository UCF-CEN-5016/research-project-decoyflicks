import numpy as np
import tensorflow as tf
from tensorflow import keras

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = 42

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        real_images = real_images[0]
        batch_size = real_images.shape[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([real_images, generated_images], axis=0)
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), seed=self.seed_generator)
        generated_images = self.generator(random_latent_vectors)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(generated_images, training=True)
            g_loss = self.loss_fn(tf.ones((batch_size, 1)), predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

    def fit(self, dataloader, epochs=1):
        for epoch in range(epochs):
            for real_images in dataloader:
                self.train_step(real_images)

discriminator = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1)
])

generator = keras.Sequential([
    keras.layers.Dense(7*7*128, input_dim=100, activation="relu"),
    keras.layers.Reshape((7, 7, 128)),
    keras.layers.BatchNormalization(momentum=0.8),
    keras.layers.UpSampling2D(),
    keras.layers.Conv2D(64, kernel_size=3, padding="same"),
    keras.layers.Activation("relu"),
    keras.layers.BatchNormalization(momentum=0.8),
    keras.layers.Conv2D(1, kernel_size=3, padding='same'),
    keras.layers.Activation("tanh")
])

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=100)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

dataloader = [[np.random.rand(32, 28, 28)]]
gan.fit(dataloader, epochs=1)