import tensorflow as tf
from tensorflow.keras import layers

# Attempt to use experimental SyncBatchNormalization
# This fails in TF 2.18.0 where experimental namespace was removed
try:
    norm_layer = layers.experimental.SyncBatchNormalization()
    print("Successfully created experimental SyncBatchNormalization")
except AttributeError as e:
    print(f"Error: {e}")
    print("This occurs in TF 2.18.0 where experimental.layers was reorganized")

# Correct alternative for TF 2.18.0+
try:
    norm_layer = layers.BatchNormalization()  # Standard alternative
    # Or for sync version:
    # norm_layer = layers.experimental.SyncBatchNormalization (new location)
    print("Successfully created standard BatchNormalization")
except Exception as e:
    print(f"Alternative also failed: {e}")