import tensorflow as tf
from tensorflow.keras.layers import ConcatV2

# Create a model with the problematic layer
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    ConcatV2(axis=1, values=[tf.zeros((10, 0)), tf.zeros((10, 0))])  # Pass length 0 tensors as predicted boxes
])

# Try to run the model
model.predict(tf.ones((10, 10)))  # Should raise a ValueError