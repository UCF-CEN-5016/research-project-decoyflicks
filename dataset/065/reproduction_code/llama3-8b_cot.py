import tensorflow as tf

try:
    from keras.utils import FeatureSpace
except ImportError as e:
    print(f"Error importing FeatureSpace: {e}")

# Minimal environment setup
print(tf.__version__)
print(tf.keras.__version__)

# Triggering condition
print("Attempting to use FeatureSpace...")