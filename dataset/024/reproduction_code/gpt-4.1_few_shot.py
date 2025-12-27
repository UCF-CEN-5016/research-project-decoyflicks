# Directory structure simulation:
# mypackage/
# ├── submodule/
# │   ├── __init__.py  (missing)
# │   └── example.py
#
# example.py contains:
# from mypackage.submodule import something  # Fails if __init__.py is missing

import os
import sys
import tempfile

# Create a temporary package folder without __init__.py
temp_dir = tempfile.TemporaryDirectory()
pkg_dir = f"{temp_dir.name}/mypackage/submodule"
os.makedirs(pkg_dir)

# Write a dummy module inside submodule
with open(f"{pkg_dir}/example.py", "w") as f:
    f.write("def foo():\n    return 'bar'\n")

# Write a script that tries to import from submodule
script = """
from mypackage.submodule import example

print(example.foo())
"""

# Add the parent directory of mypackage to sys.path
sys.path.insert(0, temp_dir.name)

try:
    # Attempt import (should fail due to missing __init__.py)
    exec(script)
except ModuleNotFoundError as e:
    print(f"Caught error: {e}")

# Now add __init__.py to submodule folder to fix the import
with open(f"{pkg_dir}/__init__.py", "w") as f:
    f.write("# Init file to make submodule a package\n")

try:
    # Retry import after adding __init__.py (should succeed)
    exec(script)
finally:
    temp_dir.cleanup()