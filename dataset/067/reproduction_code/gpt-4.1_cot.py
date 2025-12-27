import tensorflow as tf
import numpy as np

# Setup minimal environment
batch_size = 2
height = 3
width = 3
num_classes = 4

# Random logits output from model (before softmax)
logits = tf.random.uniform(
    shape=(batch_size, height, width, num_classes), minval=-1, maxval=1, dtype=tf.float32
)

# Correct labels: shape (batch_size, height, width)
labels_correct = tf.random.uniform(
    shape=(batch_size, height, width), minval=0, maxval=num_classes, dtype=tf.int32
)

# Incorrect labels: shape (batch_size, height, width, 1) -> extra dimension
labels_incorrect = tf.expand_dims(labels_correct, axis=-1)

# Setup loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Calculate loss with correct labels
loss_correct = loss_fn(labels_correct, logits)

# Calculate loss with incorrect labels
loss_incorrect = loss_fn(labels_incorrect, logits)

print(f"Loss with correct labels shape {labels_correct.shape}: {loss_correct.numpy()}")
print(f"Loss with incorrect labels shape {labels_incorrect.shape}: {loss_incorrect.numpy()}")