# Ensure KerasCV is installed by running !pip install git+https://github.com/keras-team/keras-cv -q

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

np.random.seed(42)

# Define input data dimensions
batch_size = 1000
img_height = 28
img_width = 28
channels = 1

# Generate random training data
x_train = np.random.rand(batch_size, img_height, img_width, channels)
y_train = np.random.randint(0, 10, (batch_size, 1))

# Define validation data similarly to training data
x_val = np.random.rand(batch_size, img_height, img_width, channels)
y_val = np.random.randint(0, 10, (batch_size, 1))

class MyHyperModel(keras.HyperModel):
    def build(self, hp):
        units_1 = hp.Int("units_1", min_value=10, max_value=40, step=10)
        units_2 = hp.Int("units_2", min_value=10, max_value=30, step=10)

        model = keras.Sequential([
            layers.Flatten(input_shape=(img_height, img_width, channels)),  # Add Flatten layer
            layers.Dense(units=units_1, activation='relu'),
            layers.Dense(units_2, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3])),
                      loss='mse',
                      metrics=['mae'])
        return model

    def fit(self, hp, model, x, y, validation_data):
        return model.fit(x, y,
                          epochs=hp.Int('epochs', min_value=5, max_value=20),
                          validation_data=validation_data)

hypermodel = MyHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective='val_mae',
    max_trials=2,
    executions_per_trial=1,
    overwrite=True,
    directory='results',
    project_name='custom_training'
)

try:
    tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
except Exception as e:
    print(f"Error occurred during model fitting: {e}")