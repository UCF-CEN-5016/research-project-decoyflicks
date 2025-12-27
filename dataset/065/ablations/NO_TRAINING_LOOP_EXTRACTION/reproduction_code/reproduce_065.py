import keras

# Attempt to import FeatureSpace
try:
    from keras.utils import FeatureSpace
except ImportError as e:
    print(e)

# Check the installed Keras version
print("Keras version:", keras.__version__)