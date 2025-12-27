import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

# Create a simple model with Conv2D layers
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(224, 224, 3)),
    Conv2D(64, (3, 3))
])

# Save the model using TensorFlow format (save_format='tf')
model.save("test_model", save_format='tf')

# Load weights using TensorFlow format
model.load_weights("test_model")

# Check model summary
model.summary()