try:
    from tensorflow.python.framework import tensor
except ImportError as e:
    print(f"Caught ImportError: {e}")