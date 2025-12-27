import tensorflow as tf

# Try to import tensor from tensorflow.python.framework
try:
    from tensorflow.python.framework import tensor
except ImportError as e:
    print(f"ImportError: {e}")