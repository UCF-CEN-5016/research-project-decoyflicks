import tensorflow as tf
import tf_keras.optimizers.legacy

# Attempt to subclass Optimizer from legacy module
class MyOptimizer(tf_keras.optimizers.legacy.Optimizer):
    def __init__(self, name="MyOptimizer", **kwargs):
        super().__init__(name, **kwargs)

# Instantiate the optimizer
opt = MyOptimizer()