import tensorflow as tf
from tensorflow import keras

# Minimal GAN model class mimicking the official example's train_step snippet
class GAN(keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = keras.Sequential([
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(28 * 28, activation="sigmoid"),
            keras.layers.Reshape((28, 28))
        ])
        self.discriminator = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1)
        ])
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_optimizer = keras.optimizers.Adam()
        self.d_optimizer = keras.optimizers.Adam()
        self.seed_generator = 42

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # data is expected to be a batch tensor, but here it's a list (simulate the bug)
        real_images = data
        # BUG TRIGGER: real_images is a list, so this raises AttributeError
        batch_size = real_images.shape[0]  # <-- fails if real_images is list

        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        fake_images = self.generator(random_latent_vectors)

        # Skipping full train step for brevity
        return {}

# Simulate dataset yielding batches as lists instead of tensors
def fake_dataset():
    batch = tf.random.normal((32, 28, 28))
    while True:
        yield [batch]  # Yielding a list wrapping a tensor

dataset = tf.data.Dataset.from_generator(
    fake_dataset,
    output_signature=tf.TensorSpec(shape=(32, 28, 28), dtype=tf.float32),
).take(1)

gan = GAN(latent_dim=10)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# This will raise AttributeError because the batch is a list, not a tensor
gan.fit(dataset, epochs=1)