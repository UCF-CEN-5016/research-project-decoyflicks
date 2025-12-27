# test_bugs.py

import numpy as np
from tensorflow import keras
from keras.utils import FeatureSpace

# Set up data (this part is not needed for reproducing the bug)
num_classes = 10
input_shape = (28, 28, 1)

# Attempt to use FeatureSpace
feature_space = FeatureSpace()