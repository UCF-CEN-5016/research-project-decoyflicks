import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create a dataset of images (as tensors)
data = np.random.rand(100, 32, 32, 3)

# Convert to a TensorFlow Dataset
dataset = tf.data.Dataset.from_tensor_slices(data).batch(32)

# Define generator and discriminator (simplified)
generator = keras.Sequential([
    keras.layers.Dense(256, input_shape=(100,)),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(784),
    keras.layers.Activation('tanh')
])

discriminator = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(256),
    keras.layers.LeakyReLU(),
    keras.layers.Dense(1, activation='sigmoid')
])

# Define the GAN model
class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Generate fake images
        fake_images = self.generator(random_latent_vectors)

        # Train the discriminator
        with tf.GradientTape() as tape:
            # Discriminate real images
            real_output = self.discriminator(real_images)
            # Discriminate fake images
            fake_output = self.discriminator(fake_images)
            # Compute loss
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = real_loss + fake_loss
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train the generator
        with tf.GradientTape() as tape:
            fake_output = self.discriminator(fake_images)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Compile the model
gan = GAN(generator, discriminator, latent_dim=100)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True)
)

# Train the model
gan.fit(dataset, epochs=1)