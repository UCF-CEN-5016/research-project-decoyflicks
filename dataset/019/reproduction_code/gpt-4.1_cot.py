# Minimal reproduction code for the ImportError with tensorflow_text

# First, ensure the environment has the following versions (adjust as needed):
# pip install tensorflow==2.12.0 tensorflow-text==2.12.0 absl-py

import sys

print("Python version:", sys.version)

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except ImportError as e:
    print("Failed to import TensorFlow:", e)
    raise

try:
    import tensorflow_text as text
    print("Imported tensorflow_text successfully.")
except ImportError as e:
    print("Failed to import tensorflow_text:", e)
    raise

# Optional: import tensorflow_models if this package exists and triggers error
try:
    import tensorflow_models as tfm
    print("Imported tensorflow_models successfully.")
except ImportError as e:
    print("Failed to import tensorflow_models:", e)
    # This is the original import that triggers the error
    raise