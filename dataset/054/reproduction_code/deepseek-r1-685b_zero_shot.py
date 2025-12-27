import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def load_data():
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    x_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

model = keras.Sequential([
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))