import tensorflow as tf
import numpy as np

# Simulate segmentation output and labels
batch_size = 2
height, width = 64, 64
num_classes = 3

# Model produces correct shape [batch, height, width, num_classes]
logits = tf.random.normal([batch_size, height, width, num_classes])

# Labels with incorrect extra dimension [batch, height, width, 1]
labels_wrong = np.random.randint(0, num_classes, 
                               size=(batch_size, height, width, 1))

# Correct labels shape [batch, height, width]
labels_correct = np.squeeze(labels_wrong, axis=-1)

# This will fail with shape mismatch
try:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_wrong = loss_fn(labels_wrong, logits)
    print("Loss with wrong shape:", loss_wrong.numpy())
except Exception as e:
    print("Error with wrong shape:", e)

# This works correctly
loss_correct = loss_fn(labels_correct, logits)
print("Loss with correct shape:", loss_correct.numpy())