import os
import sys
from tempfile import mkdtemp

def create_temp_dir():
    temp_dir = mkdtemp()
    os.makedirs(os.path.join(temp_dir, "delf", "python", "datasets"), exist_ok=True)
    return temp_dir

def cleanup_temp_dir(temp_dir):
    os.rmdir(os.path.join(temp_dir, "delf", "python", "datasets"))
    os.rmdir(os.path.join(temp_dir, "delf", "python"))
    os.rmdir(os.path.join(temp_dir, "delf"))
    os.rmdir(temp_dir)

def simulate_import_error(temp_dir):
    try:
        sys.path.insert(0, temp_dir)
        from delf.python.training.model import export_model_utils
        print("Import successful!")
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        print("Solution: Add __init__.py to datasets folder")
    finally:
        sys.path.remove(temp_dir)

def main():
    temp_dir = create_temp_dir()
    simulate_import_error(temp_dir)
    cleanup_temp_dir(temp_dir)

if __name__ == "__main__":
    main()