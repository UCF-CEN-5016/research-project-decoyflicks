import os

# Ensure KERAS_BACKEND is set early
os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import re
import string
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers as keras_layers
from tensorflow.keras import TextVectorization

# Optional import for ops if available in this TF version
try:
    from tensorflow.keras import ops as keras_ops
except Exception:
    keras_ops = None

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings