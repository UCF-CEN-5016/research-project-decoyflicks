import sys
import importlib.util
import importlib.machinery

# Try to import tensorflow_models as tfm
try:
    import tensorflow_models as tfm
except ImportError as e:
    print(f"ImportError: {e}")

# Try to import tensorflow_text
try:
    import tensorflow_text
except ImportError as e:
    print(f"ImportError: {e}")

# Check Python version
print(f"Python version: {sys.version}")

# Check tensorflow version
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.version.VERSION}")
except ImportError as e:
    print(f"ImportError: {e}")

# Try to import tensorflow_text.core.pybinds
try:
    import tensorflow_text.core.pybinds
except ImportError as e:
    print(f"ImportError: {e}")