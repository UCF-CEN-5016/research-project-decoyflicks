import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class GAN:
    def __init__(self, discriminator, generator, latent_dim):
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

gan = GAN(discriminator=Sequential([Dense(1)]), generator=Sequential([Dense(10)]), latent_dim=5)
gan.compile(
    d_optimizer=Adam(learning_rate=0.0003),
    g_optimizer=Adam(learning_rate=0.0003),
    loss_fn='binary_crossentropy',
)

real_images = [[1, 2], [3, 4]]
batch_size = real_images[0].shape[0]
random_latent_vectors = np.random.normal(
    shape=(batch_size, gan.latent_dim), seed=None
)
gan.fit(real_images, epochs=1)