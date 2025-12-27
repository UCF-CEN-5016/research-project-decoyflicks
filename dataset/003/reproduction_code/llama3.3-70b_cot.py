import tensorflow as tf

# Try to access the experimental SyncBatchNormalization layer
try:
    from tensorflow.keras.layers import experimental
    experimental.SyncBatchNormalization
except AttributeError as e:
    print(f"Error accessing experimental layers: {e}")

# Attempt to import and use the object detection builder
try:
    from object_detection.builders import model_builder
    print("Successfully imported model_builder")
except ImportError as e:
    print(f"Error importing model_builder: {e}")

# Minimal example to demonstrate the issue
class FreezableSyncBatchNorm:
    def __init__(self):
        try:
            self.layer = tf.keras.layers.experimental.SyncBatchNormalization()
        except AttributeError as e:
            print(f"Error creating SyncBatchNormalization layer: {e}")

# Create an instance of the custom layer
try:
    FreezableSyncBatchNorm()
except Exception as e:
    print(f"An error occurred: {e}")