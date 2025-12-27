import keras

# Attempt to import FeatureSpace
try:
    from keras.utils import FeatureSpace
except ImportError as e:
    print(e)

# Check the Keras version and path
print("Keras version:", keras.__version__)
print("Keras path:", keras.__file__)