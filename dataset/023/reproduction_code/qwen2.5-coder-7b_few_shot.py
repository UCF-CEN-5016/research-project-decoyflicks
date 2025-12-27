from typing import Optional, Any


def import_object_detection_model_lib() -> Optional[Any]:
    """
    Attempt to import model_lib from the TensorFlow Models object_detection package.

    Returns the imported module/object on success, or None on failure.
    Prints a success or failure message matching the original behavior.
    """
    try:
        from tensorflow_models.research.object_detection import model_lib
    except ImportError as err:
        print(f"Import failed: {err}")
        return None
    else:
        print("Import successful")
        return model_lib


if __name__ == "__main__":
    model_lib = import_object_detection_model_lib()