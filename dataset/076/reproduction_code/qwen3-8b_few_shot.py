import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

# Create a simple model
model = Model()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))

# Save model using TensorFlow SavedModel format (not HDF5)
model.save("coco_model", save_format="tf")  # Save with 'tf' format

# Attempt to load using HDF5 format (which is incompatible)
try:
    loaded_model = tf.keras.models.load_model("coco_model", compile=False)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)