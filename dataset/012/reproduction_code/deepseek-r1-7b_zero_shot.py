# This script demonstrates fixing an AttributeError caused by using a deprecated optimizer class.

from tensorflow.keras.optimizers import Optimizer as tf_keras_Optimizer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class ExponentialMovingAverage(tf_keras_Optimizer):
    pass  # Minimal implementation to avoid errors

model = Model()
input_layer = Input(shape=(28,28,3))
conv_layer = tf_keras_OptimizerConvLayer(input_layer)
model.add(conv_layer)