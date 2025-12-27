from object_detection.core import model_builder

# Update layer imports if necessary (e.g., for SyncBatchNormalization)
from tf.keras.layers import SyncBatchNormalization, Layer, Input, Dense, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout