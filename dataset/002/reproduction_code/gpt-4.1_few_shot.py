# This code simulates the import error encountered when running TensorFlow 2.x code that tries to import from tensorflow.contrib

try:
    from tensorflow.contrib.quantize.python import graph_matcher
except ModuleNotFoundError as e:
    print(f"Caught error: {e}")