import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import string
import re
import numpy as np

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

import keras
from keras import layers
from keras.layers import TextVectorization  # Ensure this is imported to avoid the ImportError in ops module

# This line should cause the ImportError
from keras import ops

print("This code should fail with an ImportError at 'ops' module import")