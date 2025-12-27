import tensorflow as tf

# Attempt to import a module from tensorflow.contrib
try:
    from tensorflow.contrib.quantize.python import graph_matcher
except ModuleNotFoundError as e:
    print(f"Error: {e}")

# This should raise a ModuleNotFoundError because tensorflow.contrib is not available in TensorFlow 2.x