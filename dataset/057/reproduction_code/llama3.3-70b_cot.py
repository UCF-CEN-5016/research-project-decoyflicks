# Import necessary libraries
import os

# Set the Keras backend to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import Keras and its components
import keras
from keras import layers

# Attempt to import ops from Keras, which triggers the bug
try:
    from keras import ops
except ImportError as e:
    print(f"Error importing ops from Keras: {e}")

# Import other necessary libraries
import numpy as np
import tensorflow as tf

# Print the version of Keras being used
print(f"Keras version: {keras.__version__}")