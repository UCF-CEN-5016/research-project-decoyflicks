import tensorflow_models as tfm

print("Before importing")
# This should raise an ImportError
try:
    tfm.core.pybinds.tflite_registrar  # Try to import
except ImportError as e:
    print(f"Error: {e}")

print("After importing")