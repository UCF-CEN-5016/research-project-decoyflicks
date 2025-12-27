import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic point cloud data
num_points = 1000
num_features = 3
num_classes = 5

# Create train and test datasets
train_data = np.random.rand(800, num_points, num_features)
train_labels = np.random.randint(0, num_classes, 800)
test_data = np.random.rand(200, num_points, num_features)
test_labels = np.random.randint(0, num_classes, 200)

# Simple PointNet-like model
def build_model():
    inputs = keras.Input(shape=(num_points, num_features))
    x = layers.Conv1D(64, 1, activation="relu")(inputs)
    x = layers.GlobalMaxPooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=outputs)

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# The bug: Using test set as validation set
model.fit(
    train_data, train_labels,
    epochs=5,
    validation_data=(test_data, test_labels)  # Incorrect validation set usage
)

# Proper approach would be:
# 1. Split train into train/validation
# 2. Only use test_data for final evaluation