import tensorflow as tf
from tensorflow import keras

# First show keras version
print(f"Keras version: {keras.__version__}")

# Try different import paths that might be attempted by users
try:
    print("\nTrying import from keras.utils...")
    from keras.utils import FeatureSpace
except ImportError as e:
    print(f"ImportError (keras.utils): {e}")

try:
    print("\nTrying import from tensorflow.keras.utils...")
    from tensorflow.keras.utils import FeatureSpace
except ImportError as e:
    print(f"ImportError (tensorflow.keras.utils): {e}")

# Try to create a simple feature space object to demonstrate the issue
try:
    feature_space = FeatureSpace(
        features={
            "numeric_feature": "float",
            "categorical_feature": "string"
        }
    )
except NameError as e:
    print(f"\nNameError when trying to use FeatureSpace: {e}")

print("\nThis error occurs because FeatureSpace is not available in the current version of Keras")