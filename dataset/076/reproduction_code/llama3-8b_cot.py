import tensorflow as tf
from tensorflow.keras import layers, models

# Minimal setup
physical_devices = tf.config.list_logical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Create a simple model with some layers that have weights (e.g., Conv2D)
model = models.Sequential([
    layers.Conv2D(64, (7, 7), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
])

# Define the path to load weights from
MODEL_DIR = 'path/to/model_dir'

# Load weights for this model using the `load_weights` method
model.load_weights(MODEL_DIR, by_name=True)

# This should raise a `NotImplementedError`