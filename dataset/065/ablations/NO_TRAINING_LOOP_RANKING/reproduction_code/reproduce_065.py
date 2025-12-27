import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras.utils import FeatureSpace  # This line will raise ImportError

print(keras.__version__)  # Check Keras version
print(keras.__file__)  # Check Keras installation path