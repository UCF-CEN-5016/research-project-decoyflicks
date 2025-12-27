# Import necessary libraries
import numpy as np
from tensorflow import keras
from keras_cv import layers

# Load a sample dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Create a model with a layer from kerasCV
model = keras.Sequential([
    layers.RandomRotation(factor=0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# This will cause an error due to incompatible data type
model.fit(x_train, y_train)