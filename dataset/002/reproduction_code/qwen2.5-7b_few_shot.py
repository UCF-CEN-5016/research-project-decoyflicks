import tensorflow as tf

try:
    from tensorflow.contrib.quantize.python import graph_matcher
    print("Import successful")
except ModuleNotFoundError:
    print("ModuleNotFoundError: graph_matcher is not available in TensorFlow 2.x")