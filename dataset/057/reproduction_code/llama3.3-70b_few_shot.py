import os

# Set the Keras backend to tensorflow
os.environ["KERAS_BACKEND"] = "tensorflow"

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras as tf_keras

# Attempt to import 'ops' from 'keras', which will raise an ImportError
try:
    from keras import ops
except ImportError as e:
    print(f"Error: {e}")

# Alternative import that should work
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization