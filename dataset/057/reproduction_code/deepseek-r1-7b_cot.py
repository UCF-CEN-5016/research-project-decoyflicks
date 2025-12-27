import tensorflow as tf
from tensorflow.keras import layers

# Updated import for ops if needed
from tensorflow.keras.ops import ops  # If specific Ops are required, import them here

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import string
import re
import numpy as np

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

import tensorflow.keras.layers as layers  # Or import specific layers directly if needed

# Updated imports using TensorFlow's modules instead of standalone Keras
import tensorflow as tf
from tensorflow.keras import TextVectorization