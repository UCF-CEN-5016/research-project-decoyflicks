import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import os
import numpy as np
from tensorflow import keras
import tensorflow_hub as hub

# Attempt to compile a Keras model using `model.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])`
# This should raise an ImportError: cannot import name 'ops' from 'keras'
model = keras.Sequential()
model.add(keras.layers.Dense(10, activation='relu'))
model.compile(metrics=["Accuracy", "Recall", "Precision", "AUC"])