import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf

import keras
from keras import layers
from keras import ops  # This will trigger the ImportError