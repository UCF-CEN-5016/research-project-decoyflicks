# Install dependencies
# pip install -U tensorflow_decision_forests

import tensorflow as tf
from tensorflow import keras
from keras.utils import FeatureSpace

# Attempt to import FeatureSpace to trigger the bug
try:
    from keras.utils import FeatureSpace
except ImportError as e:
    print(e)