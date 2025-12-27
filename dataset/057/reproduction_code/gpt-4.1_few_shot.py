import os

# Set Keras backend environment variable
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras import layers

# This import raises ImportError in Keras 2.14.0
from keras import ops

from keras.layers import TextVectorization

print("Imports succeeded")