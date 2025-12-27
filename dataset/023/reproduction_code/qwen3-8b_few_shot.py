# Attempting to import from non-standard TensorFlow Models path  
try:  
    from models.research.object_detection import model_lib  
    print("Import successful")  
except ImportError as e:  
    print(f"Import failed: {e}")