import tensorflow as tf
from tensorflow import keras

class DetectionFromImageModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(1)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
    def __call__(self, x):
        return self.dense(x)

model = DetectionFromImageModule()
tf.saved_model.save(model, "saved_model")
loaded = tf.saved_model.load("saved_model")
print(hasattr(loaded, "outputs"))