import tensorflow as tf

# Ensure TensorFlow and TensorFlow Text are installed
# pip install tensorflow tensorflow-text

try:
    import tensorflow_models as tfm
except ImportError as e:
    print(f"ImportError occurred: {e}")