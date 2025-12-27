import numpy as np
import tensorflow as tf
from tensorflow import keras

# Define a simple model
model = keras.Sequential([
    keras.layers.Conv2D(1, (3, 3), activation='relu', input_shape=(10, 10, 1)),
    keras.layers.Flatten(),
    keras.layers.Dense(3, activation='softmax')
])

# Compile the model with sparse categorical cross entropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Incorrect label format (last dimension should be removed)
incorrect_labels = np.random.randint(0, 3, size=(10, 10, 1))

# Correct label format
correct_labels = np.random.randint(0, 3, size=(10, 10))

# Input data
input_data = np.random.rand(1, 10, 10, 1)

# This will cause incorrect loss calculation
model.train_on_batch(input_data, incorrect_labels)
print("Loss with incorrect labels:", model.evaluate(input_data, incorrect_labels))

# This will cause correct loss calculation
model.train_on_batch(input_data, correct_labels)
print("Loss with correct labels:", model.evaluate(input_data, correct_labels))