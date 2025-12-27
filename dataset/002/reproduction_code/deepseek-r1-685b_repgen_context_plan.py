import tensorflow as tf

try:
    from tensorflow.python.quantization import graph_matcher
    print("Successfully imported graph_matcher")
except ImportError as e:
    print(f"ImportError: {e}")
    print("This occurs because the module has been moved in TensorFlow 2.x")

print(f"TensorFlow version: {tf.__version__}")
if tf.__version__.startswith('2'):
    print("Use tf.compat.v1 or migrate to native TensorFlow 2.x APIs")