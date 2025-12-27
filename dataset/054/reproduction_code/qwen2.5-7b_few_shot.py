import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(10,)))
    model.add(Dense(1))
    return model

def generate_dummy_data(num_samples, num_features):
    data = np.random.rand(num_samples, num_features)
    labels = np.random.rand(num_samples, 1)
    return data, labels

# Create dummy datasets
train_data, train_labels = generate_dummy_data(100, 10)
test_data, test_labels = generate_dummy_data(20, 10)

# Build a simple model
model = create_model()

# Train the model with validation split
model.compile(optimizer='adam', loss='mse')
model.fit(train_data, train_labels, epochs=5, validation_data=(test_data, test_labels))