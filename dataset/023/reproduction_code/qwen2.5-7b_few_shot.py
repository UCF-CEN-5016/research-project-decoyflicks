# Importing from standard TensorFlow Models path
try:
    from tensorflow_models.research.object_detection import model_lib
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")