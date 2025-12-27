import math
import urllib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf

# Attempt to import FeatureSpace to reproduce the bug
try:
    from keras.utils import FeatureSpace
except ImportError as e:
    print(e)

# Check the installed version of Keras
print(keras.__version__)