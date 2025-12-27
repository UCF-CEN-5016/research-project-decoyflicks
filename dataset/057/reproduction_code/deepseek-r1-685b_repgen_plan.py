import os

# Set Keras backend explicitly to TensorFlow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import necessary Keras components
import keras

# Check if ops module is available in Keras
ops_available = 'ops' in dir(keras)

print(f"Keras version: {keras.__version__}")
print(f"ops module available: {ops_available}")