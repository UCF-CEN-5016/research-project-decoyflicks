import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
x = layers.Conv2D(64, 3, padding="same")(x)
outputs = layers.Conv2D(3, 3, padding="same")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

x_train = np.random.random((100, 64, 64, 3))
y_train = np.random.randint(0, 3, size=(100, 64, 64, 1))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train, epochs=1)