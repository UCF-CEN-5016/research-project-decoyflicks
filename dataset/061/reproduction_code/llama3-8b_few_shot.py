import keras

# Incorrect Keras version
try:
    from keras.utils import data_utils
except ImportError:
    print("Error: cannot import name 'data_utils' from 'keras.utils'")

# Correct Keras version (required for the example to run)
print("To fix, install Keras 2.12.0 or later:")
print("pip install keras==2.12.0")