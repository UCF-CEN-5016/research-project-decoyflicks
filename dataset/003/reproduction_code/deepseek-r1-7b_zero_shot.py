import tensorflow as tf
from object_detection.builders import model_builder

# Replace layers.experimental with just layers
from tensorflow.keras.layers import SyncBatchNormalization

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 3)),
    SyncBatchNormalization(),
    tf.keras.layers.Dense(10)
])