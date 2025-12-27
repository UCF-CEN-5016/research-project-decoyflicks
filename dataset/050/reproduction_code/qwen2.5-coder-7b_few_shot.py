import torch
from torch.utils.data import DataLoader
import tensorflow as tf
from tensorflow import keras
from typing import List

class TorchDummyDataset(torch.utils.data.Dataset):
    """Dummy dataset that returns a list of tensors."""
    def __init__(self):
        # Each item is a tensor of shape (32, 28, 28)
        self._items: List[torch.Tensor] = [torch.randn(32, 28, 28) for _ in range(10)]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._items[index]


class GAN(keras.Model):
    """Simple GAN wrapper holding discriminator and generator models."""
    def __init__(self, discriminator: keras.Model, generator: keras.Model, latent_dim: int):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        # Placeholders for optimizers and loss to be set in compile()
        self.d_optimizer = None
        self.g_optimizer = None
        self.loss_fn = None

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        # Preserve names used by callers
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images: torch.Tensor):
        # Maintain original minimal behavior: compute batch size and do nothing else
        batch_size = real_images.shape[0]
        return batch_size


# Prepare dataset and dataloader
torch_dataset = TorchDummyDataset()
torch_dataloader = DataLoader(torch_dataset, batch_size=32)

# Build discriminator and generator using Keras Sequential API
discriminator = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=(784,)),
    keras.layers.Dense(1),
])

generator = keras.models.Sequential([
    keras.layers.Dense(784, input_shape=(100,)),
    keras.layers.Activation('tanh'),
])

# Instantiate and compile GAN
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=100)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Run train_step for each batch from the DataLoader
for batch in torch_dataloader:
    gan.train_step(batch)