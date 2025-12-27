import tensorflow as tf
import numpy as np

def generate_simulated_data(batch_size, height, width, num_classes):
    logits = tf.random.normal([batch_size, height, width, num_classes])
    labels_wrong = np.random.randint(0, num_classes, size=(batch_size, height, width, 1))
    labels_correct = np.squeeze(labels_wrong, axis=-1)
    return logits, labels_wrong, labels_correct

def calculate_loss(labels, logits):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss = loss_fn(labels, logits)
    return loss

# Simulate segmentation output and labels
batch_size = 2
height, width = 64, 64
num_classes = 3

logits, labels_wrong, labels_correct = generate_simulated_data(batch_size, height, width, num_classes)

# This will fail with shape mismatch
try:
    loss_wrong = calculate_loss(labels_wrong, logits)
    print("Loss with wrong shape:", loss_wrong.numpy())
except Exception as e:
    print("Error with wrong shape:", e)

# This works correctly
loss_correct = calculate_loss(labels_correct, logits)
print("Loss with correct shape:", loss_correct.numpy())