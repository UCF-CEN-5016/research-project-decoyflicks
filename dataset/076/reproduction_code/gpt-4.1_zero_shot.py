import tensorflow as tf
from tensorflow.keras import layers, models
import h5py
import numpy as np

class DummyLayer(layers.Layer):
    def build(self, input_shape):
        self.w = tf.constant(np.ones((3, 3, 3, 64), np.float32))
        self.b = tf.constant(np.ones((64,), np.float32))
    def call(self, inputs):
        return inputs

class DummyModel(models.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = DummyLayer()
    def call(self, inputs):
        return self.conv1(inputs)

model = DummyModel()
model(tf.random.uniform((1,224,224,3)))

filepath = "weights.h5"
with h5py.File(filepath, "w") as f:
    grp = f.create_group("conv1")
    grp.create_dataset("kernel:0", data=np.ones((7,7,3,64), np.float32))
    grp.create_dataset("bias:0", data=np.ones((64,), np.float32))

model.load_weights(filepath, by_name=True)