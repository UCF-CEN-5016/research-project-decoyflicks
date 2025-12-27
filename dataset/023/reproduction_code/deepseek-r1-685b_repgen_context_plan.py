import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

def import_object_detection_api():
    try:
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        print("Imports successful!")
    except ImportError as e:
        print(f"Import failed with error: {e}")
        print("This typically occurs when the TF models repo isn't properly installed")

# Uncomment and run the following setup commands to fix the import issue
"""
!git clone https://github.com/tensorflow/models.git
%cd models/research/
!protoc object_detection/protos/*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
"""

import_object_detection_api()