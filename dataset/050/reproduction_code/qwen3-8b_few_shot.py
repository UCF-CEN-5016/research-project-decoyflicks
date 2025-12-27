import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow import keras

# Dummy dataset that returns a list of tensors
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [torch.randn(32, 28, 28) for _ in range(10)]  # List of tensors
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=32)

# Keras model code
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile(optimizer=d_optimizer, loss_fn=loss_fn)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        real_images = data  # data is a list of tensors
        batch_size = real_images.shape[0]  # Error: 'list' object has no attribute 'shape'

# Create and compile the GAN
gan = GAN(
    discriminator=keras.models.Sequential([keras.layers.Dense(100, input_shape=(784,)), keras.layers.Dense(1)]),
    generator=keras.models.Sequential([keras.layers.Dense(784, input_shape=(100,)), keras.layers.Activation('tanh')]),
    latent_dim=100
)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Attempt to fit the model
gan.fit(dataloader, epochs=1)