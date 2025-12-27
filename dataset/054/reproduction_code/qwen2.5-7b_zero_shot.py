import tensorflow as tf
import numpy as np

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(10,))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Generate synthetic data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)
X_test = np.random.rand(20, 10)
y_test = np.random.rand(20, 1)

# Define and compile the model
model = build_model()

# Train the model with test set as validation data
model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test))