import tensorflow as tf
from tensorflow.keras import layers

# Check if 'experimental' attribute exists in 'keras.layers'
has_experimental = hasattr(layers, 'experimental')

print(has_experimental)