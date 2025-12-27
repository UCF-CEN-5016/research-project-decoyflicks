import numpy as np
from keras.layers import Dense
from keras.models import Sequential, Model
from keras.optimizers import Adam


class GAN:
    def __init__(self, discriminator_model: Sequential, generator_model: Sequential, latent_dim: int):
        self.discriminator = discriminator_model
        self.generator = generator_model
        self.latent_dim = latent_dim
        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def _sample_latent(self, batch_size: int):
        return np.random.normal(size=(batch_size, self.latent_dim))

    def fit(self, real_data, epochs: int = 1):
        real_array = np.array(real_data, dtype=np.float32)
        batch_size = real_array.shape[0]

        for _ in range(epochs):
            # Generate fake samples
            latent_vectors = self._sample_latent(batch_size)
            fake_samples = self.generator.predict(latent_vectors)

            # Prepare labels
            real_labels = np.ones((batch_size, 1), dtype=np.float32)
            fake_labels = np.zeros((batch_size, 1), dtype=np.float32)

            # Train discriminator on real and fake
            self.discriminator.compile(optimizer=self.d_optimizer, loss=self.loss_fn)
            combined_X = np.concatenate([real_array, fake_samples], axis=0)
            combined_y = np.concatenate([real_labels, fake_labels], axis=0)
            self.discriminator.train_on_batch(combined_X, combined_y)

            # Train generator via combined model (generator -> discriminator)
            # Freeze discriminator weights for generator training
            self.discriminator.trainable = False
            # Ensure generator is built (predict above builds it)
            combined_model = Model(inputs=self.generator.input, outputs=self.discriminator(self.generator.output))
            combined_model.compile(optimizer=self.g_optimizer, loss=self.loss_fn)
            # Generator tries to trick discriminator into predicting ones
            generator_target = np.ones((batch_size, 1), dtype=np.float32)
            combined_model.train_on_batch(latent_vectors, generator_target)
            # Unfreeze discriminator
            self.discriminator.trainable = True


# Instantiate GAN with simple Sequential models
discriminator = Sequential([Dense(1)])
generator = Sequential([Dense(10)])
gan = GAN(discriminator_model=discriminator, generator_model=generator, latent_dim=5)

# Compile optimizers and loss
gan.compile(
    d_optimizer=Adam(learning_rate=0.0003),
    g_optimizer=Adam(learning_rate=0.0003),
    loss_fn='binary_crossentropy',
)

# Example real data
real_images = [[1, 2], [3, 4]]
real_array = np.array(real_images, dtype=np.float32)

# Determine batch size and a random latent batch (as in original intent)
batch_size = real_array.shape[0]
random_latent_vectors = np.random.normal(size=(batch_size, gan.latent_dim))

# Run one epoch of training
gan.fit(real_images, epochs=1)