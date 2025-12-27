import sys
import tensorflow as tf

try:
    from object_detection import builders
    from object_detection import config_util
    from object_detection import model_lib_v2
    from object_detection import pipeline
    from object_detection import train_lib_v2
except ImportError as e:
    print(f"ImportError: {e}")

print(f"TensorFlow version: {tf.version.VERSION}")
print(f"Python version: {sys.version}")

# Try to import required libraries from tensorflow models
try:
    import tensorflow_models as tfm
except ImportError as e:
    print(f"ImportError: {e}")