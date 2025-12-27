import tensorflow as tf
from tensorflow.keras import layers

# Attempt to use experimental SyncBatchNormalization
try:
    norm_layer = layers.BatchNormalization()
    print("Successfully created BatchNormalization")
except Exception as e:
    print(f"Error: {e}")
    print("Failed to create BatchNormalization")

# Alternative for sync version in newer TensorFlow versions
try:
    norm_layer = layers.experimental.SyncBatchNormalization()
    print("Successfully created SyncBatchNormalization")
except Exception as e:
    print(f"Error: {e}")
    print("Failed to create SyncBatchNormalization")