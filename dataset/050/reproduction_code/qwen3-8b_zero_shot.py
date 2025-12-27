import torch
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

class CustomDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 1, 28, 28)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx]

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=32)

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

generator = tf.keras.Sequential([
    tf.keras.Input(shape=(100,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28, 1))
])

discriminator.compile(optimizer=Adam(learning_rate=0.0003), loss=BinaryCrossentropy())
generator.compile(optimizer=Adam(learning_rate=0.0003), loss=BinaryCrossentropy())

gan = tf.keras.models.Sequential([
    generator,
    discriminator
])

gan.compile(optimizer=Adam(learning_rate=0.0003), loss=BinaryCrossentropy())

gan.fit(dataloader, epochs=1)