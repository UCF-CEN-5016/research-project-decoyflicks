# Reproduces the import error with Keras >= 2.13.0
import keras
from keras.utils import data_utils  # Fails in Keras 2.13.0+

print("Keras version:", keras.__version__)
print("data_utils imported successfully")

# Expected behavior (works in Keras 2.12.0):
# Should import successfully and print version

# Actual behavior (in Keras 2.13.0+):
# ImportError: cannot import name 'data_utils' from 'keras.utils'