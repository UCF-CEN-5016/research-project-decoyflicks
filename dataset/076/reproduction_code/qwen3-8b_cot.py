import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D

# Create a simple model with Conv2D layers
model = Model()
model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3)))

# Save the model using TensorFlow format (save_format='tf')
model.save("test_model", save_format='tf')

# Attempt to load weights using HDF5 format (default)
# This will trigger the NotImplementedError
model.load_weights("test_model", by_name=True)