import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Create a simple Conv2D model
def create_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, (7, 7), padding='same', name='conv1')(inputs)
    model = keras.Model(inputs, x)
    return model

model = create_model()

# Directory to save weights in TF format (NOT h5)
save_dir = './tf_weights'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save weights in TF format, NOT h5
# This creates a directory with TF checkpoint files
model.save_weights(save_dir, save_format='tf')

# Now try to load weights specifying the directory and by_name True
# This should raise NotImplementedError because load_weights expects a h5 file here
try:
    model.load_weights(save_dir, by_name=True)
except NotImplementedError as e:
    print("Caught NotImplementedError as expected:")
    print(e)