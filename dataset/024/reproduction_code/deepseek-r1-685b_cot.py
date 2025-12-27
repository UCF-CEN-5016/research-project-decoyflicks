# Reproduces ModuleNotFoundError: No module named 'delf.python.datasets'

# 1. Setup - create mock package structure without __init__.py in datasets/
import os
import tempfile
import sys

# Create temporary directory structure
with tempfile.TemporaryDirectory() as tmp_dir:
    # Create package structure
    os.makedirs(os.path.join(tmp_dir, "delf", "python", "datasets"))
    
    # Create __init__.py in parent directories
    with open(os.path.join(tmp_dir, "delf", "__init__.py"), "w") as f:
        f.write("")
    with open(os.path.join(tmp_dir, "delf", "python", "__init__.py"), "w") as f:
        f.write("")
    
    # Add to Python path
    sys.path.insert(0, tmp_dir)
    
    # 2. Trigger the error
    try:
        from delf.python.datasets import mock_module
    except ModuleNotFoundError as e:
        print(f"Reproduced error: {e}")
        print("This occurs because there's no __init__.py in the datasets directory")
        
    # 3. Fix by adding __init__.py
    with open(os.path.join(tmp_dir, "delf", "python", "datasets", "__init__.py"), "w") as f:
        f.write("")
    
    # Verify fix works
    try:
        from delf.python.datasets import mock_module
        print("After adding __init__.py, the import works (though mock_module doesn't exist)")
    except ModuleNotFoundError as e:
        if "mock_module" in str(e):
            print("Import path now works, though the specific module doesn't exist")