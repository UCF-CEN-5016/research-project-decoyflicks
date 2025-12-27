import tensorflow as tf

# Minimal environment setup (just for testing)
try:
    from tensorflow.python.framework import tensor  # Trigger the bug
except ImportError as e:
    print(f"Error: {e}")

print("Running test...")