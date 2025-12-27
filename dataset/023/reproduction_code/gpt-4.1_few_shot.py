import sys
import os

# Attempt to import object_detection directly without setup
try:
    from object_detection.utils import visualization_utils as vis_util
except ModuleNotFoundError as e:
    print(f"ImportError: {e}")

# Minimal reproduction of the import error seen in the notebook:
# The object_detection module is not found because the research path is not added to sys.path.

# To reproduce the bug:
# 1. Clone the tensorflow/models repo (not shown here for brevity)
# 2. Run this code without adding the research directory to sys.path or installing dependencies

# Example:
# sys.path does not include the path to 'models/research' where object_detection resides
print("Current sys.path entries:")
for p in sys.path:
    print(p)

# This will raise ModuleNotFoundError:
# from object_detection.utils import visualization_utils as vis_util