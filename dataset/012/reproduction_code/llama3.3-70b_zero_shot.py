import tensorflow as tf
from tensorflow import keras

class ExponentialMovingAverage(keras.optimizers.legacy.Optimizer):
    pass

try:
    ExponentialMovingAverage()
except AttributeError as e:
    print(e)