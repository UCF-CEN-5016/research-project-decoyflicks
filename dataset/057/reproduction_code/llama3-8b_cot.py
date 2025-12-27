import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np

# Trigger the bug by attempting to import 'ops' from 'keras'
try:
    from keras import ops
except ImportError as e:
    print(f"Error importing 'ops' from 'keras': {e}")

print("Reproduction code executed successfully")