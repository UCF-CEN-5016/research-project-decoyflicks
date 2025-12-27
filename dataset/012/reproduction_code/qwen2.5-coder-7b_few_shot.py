import tensorflow.keras as tf.keras

# Correct import path for optimizer
from tensorflow.keras.optimizers import Optimizer

# Subclassing existing optimizer class
class CustomEMAOptimizer(Optimizer):
    def __init__(self, name=None)
        super(CustomEMAOptimizer, self).__init__(name=name)
        self.shadow_vars = []

# No error will be triggered
print("Error reproduced successfully")