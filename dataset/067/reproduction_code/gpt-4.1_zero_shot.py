import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

x = np.random.rand(2, 4, 4, 3).astype(np.float32)
y = np.random.randint(0, 5, size=(2, 4, 4, 1))

model = models.Sequential([
    layers.Conv2D(5, 3, padding='same'),
    layers.Reshape((4*4, 5)),
    layers.Softmax(axis=-1)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy()
)

model.fit(x, y, epochs=1)