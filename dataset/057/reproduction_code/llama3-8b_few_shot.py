import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import string
import re
import numpy as np

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

import keras  # Import Keras

# This line should not throw an error, but it does
from keras import ops  # Try to import Keras ops