import tensorflow as tf
from keras_tuner import Objective, RandomSearch

# Define x_train and y_train with shapes (1000, 28, 28, 1) and (1000, 1)
x_train = ...
y_train = ...

# Define x_val and y_val with shapes (1000, 28, 28, 1) and (1000, 1)
x_val = ...
y_val = ...

# Initialize MyHyperModel class as defined in keras_tuner_custom_tuner.py
class MyHyperModel:
    def build_model(self, hp):
        # Define the 2 hyperparameters for the units in dense layers
        units_1 = hp.Int("units_1", 10, 40, step=10)
        units_2 = hp.Int("units_2", 10, 30, step=10)

        # Define the model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=units_1, input_shape=(28*28,)),
            tf.keras.layers.Reshape((28, 28)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=units_2),
            tf.keras.layers.Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit_model(self, model):
        # Fit the model to x_train, y_train with batch_size=64 for 2 epochs
        history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
        return history.history['loss']

# Create a RandomSearch tuner instance with objective 'my_metric' to minimize, max_trials=2, directory='results', project_name='custom_training'
tuner = RandomSearch(
    MyHyperModel().build_model,
    objective=Objective('val_loss', direction='min'),
    max_trials=2,
    directory='results',
    project_name='custom_training'
)

# Call the search method on the tuner with x_train, y_train and validation_data=(x_val, y_val)
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Retrieve the best hyperparameters from the tuner
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Create a model instance using the best hyperparameters obtained in the previous step
model = MyHyperModel().build_model(best_hps)

# Fit the model to x_train, y_train with batch_size=64 for 2 epochs as defined in MyHyperModel.fit() method
history = MyHyperModel().fit_model(model)

# Monitor and record the training loss during fitting
losses = history

# Assert that at least one of the recorded losses is NaN or infinite
import numpy as np
assert not any(np.isnan(loss) or np.isinf(loss) for loss in losses)