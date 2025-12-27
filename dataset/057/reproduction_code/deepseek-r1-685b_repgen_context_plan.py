import os

# Set Keras backend explicitly
os.environ["KERAS_BACKEND"] = "tensorflow"

# Attempt to import TensorFlow and Keras components
import tensorflow as tf
from tensorflow import keras

print(f"Keras version: {tf.keras.__version__}")
print(f"ops module available: {'ops' in dir(tf.keras)}")