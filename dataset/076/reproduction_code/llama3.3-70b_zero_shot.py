import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Conv2D(keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(Conv2D, self).__init__()
        self.conv = keras.layers.Conv2D(filters, kernel_size)

    def call(self, inputs):
        return self.conv(inputs)

class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = Conv2D(64, (7, 7))

    def call(self, inputs):
        return self.conv(inputs)

model = Model()
model.compile(optimizer='adam', loss='mean_squared_error')

x = np.random.rand(1, 224, 224, 3)
y = np.random.rand(1, 224, 224, 64)

model.fit(x, y, epochs=1)

model.save('model.h5')

loaded_model = keras.models.load_model('model.h5', compile=False)

x = np.random.rand(1, 224, 224, 3)
y_pred = loaded_model.predict(x)