import torch
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Custom Dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create DataLoader using Custom Dataset
dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32)

# Discriminator model
discriminator = tf.keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Generator model
generator = tf.keras.Sequential([
    Input(shape=(100,)),
    Dense(256, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28, 1))
])

# Compile discriminator and generator models
optimizer = Adam(learning_rate=0.0003)
loss = BinaryCrossentropy()

discriminator.compile(optimizer=optimizer, loss=loss)
generator.compile(optimizer=optimizer, loss=loss)

# GAN model
gan = tf.keras.models.Sequential([generator, discriminator])
gan.compile(optimizer=optimizer, loss=loss)

# Train GAN model
gan.fit(dataloader, epochs=1)