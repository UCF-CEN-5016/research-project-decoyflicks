import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np  # Import numpy for dataset generation

def build_model(hp):
    units_1 = hp.Int("units_1", min_value=10, max_value=40, step=10)
    units_2 = hp.Int("units_2", min_value=10, max_value=30, step=10)
    model = keras.Sequential([
        layers.Dense(units=units_1, input_shape=(28*28,), activation='relu'),  # Added activation for better performance
        layers.Dense(units=units_2, activation='relu'),
        layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def fit_model():
    x_train = np.random.rand(1000, 28*28) * 255  # Scale to 0-255 range for better performance
    y_train = np.random.rand(1000, 1)
    x_val = np.random.rand(1000, 28*28) * 255
    y_val = np.random.rand(1000, 1)

    objective = kt.Objective('val_loss', direction='min')
    tuner = kt.RandomSearch(
        build_model,
        objective=objective,
        max_trials=2,
        directory='results',
        project_name='custom_training'
    )

    try:
        tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    except Exception as e:
        print(f"Error during model fitting: {e}")

fit_model()