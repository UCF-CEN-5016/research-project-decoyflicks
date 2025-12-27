import tf_keras

# Incorrect import path for legacy optimizer
from tf_keras.optimizers.legacy import Optimizer  # This will raise AttributeError

# Attempt to subclass non-existent Optimizer class
class CustomEMAOptimizer(Optimizer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.shadow_vars = []

# This will trigger the AttributeError
print("Error reproduced successfully")