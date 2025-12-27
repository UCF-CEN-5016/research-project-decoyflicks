import os
import sys
from tempfile import mkdtemp
from importlib.machinery import SourceFileLoader

# Create temporary directory structure
temp_dir = mkdtemp()
os.makedirs(os.path.join(temp_dir, "delf", "python", "datasets"), exist_ok=True)

# Create empty __init__.py in datasets folder to fix the error
init_py_path = os.path.join(temp_dir, "delf", "python", "datasets", "__init__.py")
open(init_py_path, 'w').close()

# Simulate the problematic import chain
try:
    sys.path.insert(0, temp_dir)
    
    # Import the module dynamically
    dataset_module = SourceFileLoader("dataset_module", init_py_path).load_module()
    export_model_utils = dataset_module.export_model_utils
    print("Import successful!")
except Exception as e:
    print(f"Error: {e}")
    print("Solution: Add __init__.py to datasets folder")
finally:
    # Cleanup
    sys.path.remove(temp_dir)
    os.remove(init_py_path)