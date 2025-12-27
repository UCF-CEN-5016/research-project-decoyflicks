import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the discriminator model
def define_discriminator(image_shape):
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5,5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0003))
    return model

# Define the generator model
def define_generator(latent_dim):
    model = keras.Sequential()
    n_nodes = 7 * 7 * 128
    model.add(layers.Dense(n_nodes, input_dim=latent_dim))
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1, 1), padding='same'))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2, 2), padding='same'))
    model.add(layers.Activation("relu"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2, 2), padding='same', activation='tanh'))
    return model

# Define the combined GAN model
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seed_generator = np.random.randint(0, 1000)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data
        real_images, _ = data

        # Sample random points in the latent space
        batch_size = real_images.shape[0] # Error occurs here
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode them to fake images
        fake_images = self.generator(random_latent_vectors, training=True)

        # Combine them with real images
        combined_images = tf.concat([real_images, fake_images], axis=0)

        # Create labels for the combined images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # Decode them to fake images
        fake_images = self.generator(random_latent_vectors, training=True)

        # Create labels for the fake images
        labels = tf.ones((batch_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Define the image shape
image_shape = (28, 28, 1)

# Define the latent dimension
latent_dim = 100

# Define the discriminator and generator models
discriminator = define_discriminator(image_shape)
generator = define_generator(latent_dim)

# Define the GAN model
gan = GAN(discriminator, generator, latent_dim)

# Compile the GAN model
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Create a sample dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.normal([100, 28, 28, 1]), tf.random.normal([100, 28, 28, 1])))
train_dataset = train_dataset.batch(32)

# Train the GAN model
gan.fit(train_dataset)