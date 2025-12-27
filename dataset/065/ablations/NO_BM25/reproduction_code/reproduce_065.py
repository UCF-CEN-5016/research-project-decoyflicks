# Install TensorFlow version 2.7 or higher
# !pip install tensorflow==2.7

import math
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf

try:
    from keras.utils import FeatureSpace
except ImportError as e:
    print(e)
    print("Error message contains:", str(e))
    print("Keras version:", keras.__version__)