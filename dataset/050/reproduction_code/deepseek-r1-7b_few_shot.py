import tensorflow as tf
from tensorflow.keras import layers, Model

# Define a simple model
model = tf.keras.Sequential([
    layers.Dense(10, input_shape=(5,))
])

# Load MNIST dataset and convert images to lists (simulating the issue)
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Convert real_images to a list of numpy arrays instead of tensors
real_images = x_train.tolist()  # Now `real_images` is a list

try:
    batch_size = len(real_images)
    print(f"Batch size: {batch_size}")
except AttributeError as e:
    print(f"Error: {e}")