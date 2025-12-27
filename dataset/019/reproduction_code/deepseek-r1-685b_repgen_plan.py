# Minimal reproduction of TensorFlow Text import error
# Requires TensorFlow and TensorFlow Text installed
try:
    import tensorflow_text as text  # Triggers the import error
    print("Successfully imported tensorflow_text")
except ImportError as e:
    print(f"Import failed with error: {e}")
    print("This occurs due to version mismatch between TensorFlow and TensorFlow Text")
    print("The undefined symbol comes from incompatible abseil versions")