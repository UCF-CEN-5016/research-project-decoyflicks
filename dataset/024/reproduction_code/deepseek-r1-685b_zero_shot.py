import os
import sys
from importlib import import_module

# Create minimal directory structure
os.makedirs("delf/python/datasets", exist_ok=True)
os.makedirs("delf/python/training/model", exist_ok=True)

# Create empty __init__.py files
with open("delf/__init__.py", "w") as f:
    f.write("from delf.python.training import model")

with open("delf/python/__init__.py", "w") as f:
    pass

with open("delf/python/training/__init__.py", "w") as f:
    f.write("from delf.python.training.model import export_model_utils")

with open("delf/python/training/model/__init__.py", "w") as f:
    f.write("from delf.python.training.model import export_model_utils")

with open("delf/python/training/model/export_model_utils.py", "w") as f:
    f.write("from delf.python.datasets.google_landmarks_dataset import googlelandmarks as gld")

# Add to path and try to import
sys.path.insert(0, os.getcwd())
import delf