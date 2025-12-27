# Attempt to import from deprecated tensorflow.contrib
try:
    from tensorflow.contrib.quantize.python import graph_matcher
    print("Successfully imported graph_matcher")
except ImportError as e:
    print(f"ImportError: {e}")
    print("This occurs because tensorflow.contrib was removed in TF 2.x")

# Suggested alternative for TF 2.x
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    if tf.__version__.startswith('2'):
        print("Use tf.compat.v1 or migrate to native TF 2.x APIs")
except ImportError:
    print("TensorFlow not installed")