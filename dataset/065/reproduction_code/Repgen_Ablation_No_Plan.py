import keras.utils

# Attempt to import FeatureSpace from keras.utils
try:
    from keras.utils import FeatureSpace
except ImportError:
    print("Error importing FeatureSpace from keras.utils")