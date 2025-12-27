import tensorflow as tf
from object_detection.core import freezable_sync_batch_norm

# This will trigger the error
print(freezable_sync_batch_norm.FreezableSyncBatchNorm)

# Change from:
# tf.keras.layers.experimental.SyncBatchNormalization
# To:
tf.keras.layers.BatchNormalization  # or tf.keras.layers.experimental.SyncBatchNormalization for older TF