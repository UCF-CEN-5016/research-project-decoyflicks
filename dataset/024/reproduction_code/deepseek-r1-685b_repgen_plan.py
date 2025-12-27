import os
import sys
import tempfile
from importlib.util import find_spec

# Create temporary directory structure
temp_dir = tempfile.mkdtemp()
os.makedirs(os.path.join(temp_dir, "delf", "python", "datasets"), exist_ok=True)

# Simulate the problematic import chain
try:
    sys.path.insert(0, temp_dir)
    
    # Check if the directory structure contains an __init__.py file
    if find_spec("delf.python.datasets") is None:
        with open(os.path.join(temp_dir, "delf", "python", "datasets", "__init__.py"), 'w'):
            pass
    
    # Attempt the same import chain that fails in DELF
    from delf.python.training.model import export_model_utils
    print("Import successful!")
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Solution: Add __init__.py to datasets folder")
finally:
    # Cleanup
    sys.path.remove(temp_dir)