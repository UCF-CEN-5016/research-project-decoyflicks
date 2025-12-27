import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Set up minimal environment
tf.random.set_seed(42)

# Create synthetic data
batch_size = 2
height, width = 4, 4
num_classes = 3

# Correct labels shape: (batch, height, width)
correct_labels = np.random.randint(0, num_classes, (batch_size, height, width))

# Incorrect labels shape (with extra dimension): (batch, height, width, 1)
incorrect_labels = np.expand_dims(correct_labels, -1)

# Simple model that outputs class probabilities
inputs = layers.Input(shape=(height, width, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
outputs = layers.Conv2D(num_classes, 3, activation='softmax')(x)
model = models.Model(inputs, outputs)

# Test with both label formats
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

print("Testing with correct label shape (h, w):")
try:
    loss = loss_fn(correct_labels, model(np.random.rand(batch_size, height, width, 3)))
    print(f"Loss computed successfully: {loss.numpy()}")
except Exception as e:
    print(f"Error with correct shape: {e}")

print("\nTesting with incorrect label shape (h, w, 1):")
try:
    loss = loss_fn(incorrect_labels, model(np.random.rand(batch_size, height, width, 3)))
    print(f"Loss computed successfully: {loss.numpy()}")
except Exception as e:
    print(f"Error with incorrect shape: {e}")