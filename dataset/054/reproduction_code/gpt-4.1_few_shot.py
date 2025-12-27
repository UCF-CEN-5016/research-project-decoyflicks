import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate dummy data
num_samples = 1000
num_features = 20
num_classes = 3

X = np.random.rand(num_samples, num_features).astype(np.float32)
y = keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples))

# Split data
train_size = 700
val_size = 150
test_size = 150

X_train = X[:train_size]
y_train = y[:train_size]

# Incorrectly using test data as validation data
X_val = X[train_size + val_size:]  # This should be separate validation set, but it's actually test set
y_val = y[train_size + val_size:]

# Normally, validation should be separate from test. Here test set is leaked as validation.
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_features,)),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Bug: validation_data is actually test data, causing data leakage
model.fit(X_train, y_train,
          epochs=5,
          validation_data=(X_val, y_val))