import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create dummy datasets
train_data = np.random.rand(100, 10)
train_labels = np.random.rand(100, 1)
test_data = np.random.rand(20, 10)
test_labels = np.random.rand(20, 1)

# Build a simple model
model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(1)
])

# Incorrectly using test data as validation
model.fit(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))