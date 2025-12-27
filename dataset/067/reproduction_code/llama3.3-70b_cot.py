# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Set up minimal environment
np.random.seed(0)
tf.random.set_seed(0)

# Define model architecture
model = keras.Sequential([
    layers.InputLayer(input_shape=(10, 10, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Output layer with 3 classes
])

# Compile model with SparseCategoricalCrossentropy loss
model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

# Generate sample data
X_train = np.random.rand(100, 10, 10, 3)  # Input shape: (batch_size, height, width, channels)
y_train = np.random.randint(0, 3, size=(100, 10, 10, 1))  # Incorrectly formatted labels with extra dimension

# Train model with incorrectly formatted labels
model.fit(X_train, y_train, epochs=10)

# To fix the bug, remove the extra dimension from the labels
y_train_correct = np.squeeze(y_train, axis=-1)  # Correctly formatted labels

# Train model with correctly formatted labels
model.fit(X_train, y_train_correct, epochs=10)