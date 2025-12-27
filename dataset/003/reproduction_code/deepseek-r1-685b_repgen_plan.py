import tensorflow as tf
from tensorflow.keras import layers

# Attempt to use SyncBatchNormalization
try:
    norm_layer = layers.SyncBatchNormalization()  # Use SyncBatchNormalization directly
    print("Successfully created SyncBatchNormalization")
except AttributeError as e:
    print(f"Error: {e}")
    print("This may happen if SyncBatchNormalization is not available in the current TensorFlow version")

# Fallback to standard BatchNormalization if SyncBatchNormalization is not available
try:
    norm_layer = layers.BatchNormalization()
    print("Successfully created standard BatchNormalization")
except Exception as e:
    print(f"Failed to create any form of BatchNormalization: {e}")