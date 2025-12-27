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
from keras.layers import TextVectorization

try:
    from keras import ops
except ImportError as e:
    print(e)