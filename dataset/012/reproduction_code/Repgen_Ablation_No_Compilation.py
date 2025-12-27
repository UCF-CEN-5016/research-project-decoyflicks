import tensorflow as tf

# Attempt to import Optimizer from tf_keras.optimizers.legacy
from tf_keras.optimizers.legacy import Optimizer as LegacyOptimizer

def test_optimizer():
    # This function is expected to use the LegacyOptimizer but will raise an error due to the bug
    optimizer = LegacyOptimizer(learning_rate=0.001, momentum=0.9)
    print(optimizer)

# Run the test function
test_optimizer()