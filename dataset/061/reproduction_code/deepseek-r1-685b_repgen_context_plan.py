import keras
from keras.utils import data_utils as keras_data_utils

print("Keras version:", keras.__version__)
print("data_utils imported successfully")

# Expected behavior (works in Keras 2.12.0):
# Should import successfully and print version

# Actual behavior (in Keras 2.13.0+):
# ImportError: cannot import name 'data_utils' from 'keras.utils'