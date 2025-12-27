import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the model
model = keras.Sequential([
    layers.InputLayer(input_shape=(10,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Generate some dummy data
np.random.seed(0)
train_dataset = np.random.rand(100, 10)
test_dataset = np.random.rand(20, 10)

# Train the model using test set as validation set
model.fit(train_dataset, train_dataset, epochs=20, validation_data=(test_dataset, test_dataset))