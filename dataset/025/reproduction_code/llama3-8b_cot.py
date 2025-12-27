import tensorflow_models as tfm
import tensorflow as tf

# Minimal environment setup
print("TensorFlow version:", tf.__version__)
print("Python version:", tfm.__version__)

try:
    # Triggering condition: try accessing RandAugment() function
    tfm.vision.augment.RandAugment()
except AttributeError as e:
    print(f"Error: {e}")