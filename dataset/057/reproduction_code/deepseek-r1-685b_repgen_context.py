import os

os.environ["KERAS_BACKEND"] = "tensorflow"  # Explicit backend setting

# Attempt to import TensorFlow
import tensorflow as tf

# Check if TensorFlow is imported successfully
if tf.__version__:
    print(f"TensorFlow version: {tf.__version__}")

# Attempt to import Keras components
from tensorflow import keras
from tensorflow.keras import backend

print(f"Keras version: {keras.__version__}")
print(f"backend module available: {'backend' in dir(keras)}")