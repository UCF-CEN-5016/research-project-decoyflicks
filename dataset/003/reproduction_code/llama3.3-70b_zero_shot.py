import tensorflow as tf
from tensorflow import keras
from object_detection.builders import model_builder

class FreezableSyncBatchNorm(keras.layers.SyncBatchNormalization):
    pass

try:
    _ = FreezableSyncBatchNorm()
except AttributeError as e:
    print(e)