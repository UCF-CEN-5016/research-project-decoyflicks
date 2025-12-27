import tensorflow as tf
from tensorflow import keras

# Define a simple Conv2D model
model = keras.models.Sequential([
    keras.layers.Conv2D(64, (7, 7), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Try to save and load the model with incompatible weight type
try:
    model.save('model.h5')
    loaded_model = keras.models.load_model('model.h5')
except NotImplementedError as e:
    print(f"Error: {e}")

# Correct way to save and load the model
model.save('model.tf', save_format='tf')
loaded_model = keras.models.load_model('model.tf')