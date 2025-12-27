# Import required libraries
import sys
import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Attempt to import object detection libraries
try:
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    print("Object detection libraries imported successfully")
except ImportError as e:
    print(f"Import error: {e}")

# Try to import specific module that causes the error
try:
    import tensorflow_models
    print("TensorFlow models imported successfully")
except ImportError as e:
    print(f"Import error: {e}")