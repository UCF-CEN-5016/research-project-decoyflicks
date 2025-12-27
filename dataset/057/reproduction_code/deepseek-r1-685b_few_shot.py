import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Explicit backend setting

# Attempt to import Keras components
import keras
from keras import ops  # This will fail in Keras 2.14.0

print(f"Keras version: {keras.__version__}")
print(f"ops module available: {'ops' in dir(keras)}")