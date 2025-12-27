import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

def build_model(hp):
    # Define the 2 hyperparameters for the units in dense layers
    units_1 = hp.Int("units_1", 10, 40, step=10)
    units_2 = hp.Int("units_2", 10, 30, step=10)

    # Define the model
    model = keras.Sequential([
        layers.Dense(units=units_1, input_shape=(28 * 28,)),
        layers.Dense(units=units_2),
        layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def custom_loss(y_true, y_pred):
    # Example of a custom loss function that might lead to NaN values
    return keras.backend.mean(keras.backend.square(y_true - y_pred))

# Setup the data
batch_size = 32
height, width = 28, 28
input_shape = (height * width)

# Create random uniform input data with shape (1000, 784)
x_train = np.random.rand(1000, input_shape)
y_train = keras.utils.to_categorical(np.random.randint(0, 10, size=(1000,)), num_classes=10)

# Define a validation dataset
x_val = np.random.rand(batch_size, input_shape)
y_val = keras.utils.to_categorical(np.random.randint(0, 10, size=(batch_size,)), num_classes=10)

# Initialize an instance of 'MyHyperModel' class (Assuming it's defined in the main file or module snippets)
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')

# Start the search for hyperparameters
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[custom_loss])

# Retrieve the best hyperparameters and model from the tuner
best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)

best_model = tuner.get_best_models()[0]
best_model.summary()