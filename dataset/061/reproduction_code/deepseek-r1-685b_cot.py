# Reproduction code for Keras data_utils import error
try:
    # This will fail in Keras versions where data_utils was moved/removed
    from keras.utils import data_utils
    print("Import successful! Using compatible Keras version.")
except ImportError as e:
    print(f"Import failed with error: {e}")
    print("This occurs in certain Keras versions due to internal reorganization.")
    print("Try: pip install keras==2.12.0")

import keras
print(keras.__version__)

import inspect
import keras.utils
print(inspect.getfile(keras.utils))

import os
keras_utils_path = inspect.getfile(keras.utils)
print(os.listdir(os.path.dirname(keras_utils_path)))