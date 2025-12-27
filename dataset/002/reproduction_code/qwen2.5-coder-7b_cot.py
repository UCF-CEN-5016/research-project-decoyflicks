import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from object_detection exporting functions from TensorFlow 2.x equivalents.

# Replace deprecated contrib imports with updated versions or direct tf imports
from tensorflow.python.util import tf2 as contrib  # If applicable

# Directly use tf imports for new functionalities
from tensorflow.keras import layers, models, utils