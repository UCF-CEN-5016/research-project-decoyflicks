import os
import sys

# Minimal environment to reproduce the issue
sys.path.insert(0, '.')  # Add current directory to Python path

from fairseq.infer import Inferer  # Import the problematic module
infer = Inferer()  # Try to create an instance of the module

print("Inferencer created:", infer)  # This should raise a ModuleNotFoundError