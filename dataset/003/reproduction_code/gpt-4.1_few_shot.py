import tensorflow as tf

# Attempting to use SyncBatchNormalization from the experimental namespace
try:
    # This line simulates the problematic import/usage from object_detection code
    class FreezableSyncBatchNorm(tf.keras.layers.experimental.SyncBatchNormalization):
        pass
except AttributeError as e:
    print(f"Caught AttributeError: {e}")