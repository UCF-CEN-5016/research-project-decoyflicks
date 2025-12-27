import tensorflow as tf
import numpy as np

# Simulate segmentation output and labels
batch_size = 2
height, width = 64, 64
num_classes = 3

# Model produces correct shape [batch, height, width, num_classes]
logits = tf.random.normal([batch_size, height, width, num_classes])

# Labels with correct shape [batch, height, width]
labels = np.random.randint(0, num_classes, size=(batch_size, height, width))

# Calculate loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(labels, logits)
print("Loss:", loss.numpy())