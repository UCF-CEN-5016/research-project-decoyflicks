# Simulates the module structure that causes the import error
import os
import sys
from tempfile import mkdtemp

# Create temporary directory structure
temp_dir = mkdtemp()
os.makedirs(os.path.join(temp_dir, "delf", "python", "datasets"), exist_ok=True)

# Simulate the problematic import chain
try:
    sys.path.insert(0, temp_dir)
    
    # Create empty __init__.py in datasets folder to fix the error
    # Comment out this line to reproduce the original error
    # open(os.path.join(temp_dir, "delf", "python", "datasets", "__init__.py"), 'w').close()
    
    # Attempt the same import chain that fails in DELF
    from delf.python.training.model import export_model_utils
    print("Import successful!")
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Solution: Add __init__.py to datasets folder")
finally:
    # Cleanup
    sys.path.remove(temp_dir)