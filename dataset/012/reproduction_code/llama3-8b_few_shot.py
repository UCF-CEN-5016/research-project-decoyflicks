import tensorflow as tf
from tf_keras.optimizers import legacy

# Try to access non-existent attribute
optimizer = legacy.Optimizer()

print(optimizer)  # Should raise AttributeError