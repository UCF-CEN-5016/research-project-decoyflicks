import tensorflow as tf
from tensorflow.keras import layers

# Define a model with standard layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
])

# Save the weights in H5 format (this works normally)
model.save_weights('custom_layer.h5')

try:
    # Attempt to load back into an identical model; this will raise the error
    loaded_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
    ])
    loaded_model.load_weights('custom_layer.h5')
except Exception as e:
    print(f"Error: {e}")

import tensorflow as tf

# If possible, load using `load_model` and then extract layers if needed
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
])
model.load_weights('custom_layer.h5')

# Alternatively, define new model structure after loading weights manually
loaded_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu'),
])
loaded_model.set_weights(model.get_weights())