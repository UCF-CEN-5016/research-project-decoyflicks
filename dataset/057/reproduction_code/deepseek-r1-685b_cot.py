# Reproduction code for Keras ops import error
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Same as original setup

import keras
from keras import ops  # This is the failing import