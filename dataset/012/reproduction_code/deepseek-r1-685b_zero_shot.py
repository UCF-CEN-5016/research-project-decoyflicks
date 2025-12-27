import tf_keras
from tf_keras.optimizers.legacy import Optimizer

class TestOptimizer(Optimizer):
    def __init__(self):
        super().__init__("test")

try:
    optimizer = TestOptimizer()
except Exception as e:
    print(f"Error: {e}")