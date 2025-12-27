import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set up minimal environment
model = keras.Sequential([
    layers.Conv2D(64, (7, 7), activation='relu', input_shape=(224, 224, 3))
])

# Save weights to HDF5 file
model.save_weights('model.h5')

# Try to load weights from HDF5 file
try:
    model.load_weights('model.h5', by_name=True)
except NotImplementedError as e:
    print(e)