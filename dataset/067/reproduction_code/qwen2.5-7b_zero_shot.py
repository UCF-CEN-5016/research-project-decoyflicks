import tensorflow as tf
import numpy as np

def create_dummy_data():
    x = np.random.rand(1, 2, 2, 1)
    y = np.random.randint(0, 3, (1, 2, 2))
    return x, y

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2, 2, 1)),
        tf.keras.layers.Conv2D(3, (1, 1)),
    ])
    return model

def train_model(model, x, y):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(x, y, epochs=1)

# 1. Create dummy input and labels
x, y = create_dummy_data()

# 2. Build model
model = build_model()

# 3. Train the model
train_model(model, x, y)