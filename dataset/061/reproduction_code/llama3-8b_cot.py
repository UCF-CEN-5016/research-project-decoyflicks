# 1. Start with necessary imports
import tensorflow as tf
from keras.utils import *  # This line is relevant for the bug

# 2. Set up minimal environment
print("Keras version:", tf.keras.__version__)
print("TensorFlow version:", tf.__version__)

try:
    from keras.utils import data_utils  # This should fail with an ImportError
except ImportError as e:
    print(f"Error importing 'data_utils' from 'keras.utils': {e}")

# 3. Add triggering conditions (none needed, just the above import statement)