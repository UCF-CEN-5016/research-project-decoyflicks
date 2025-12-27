import os
os.environ['Keras_Backend'] = 'tensorflow'

import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

batch_size = 32

# Import necessary modules and classes from the main file
from bert_classifier import BertClassifier
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy('mixed_float16')

classifier = BertClassifier(preset='bert_tiny_en_uncased_sst2')

try:
    predictions = classifier.predict(['I love modular workflows in keras-nlp'])
except AttributeError as e:
    error_output = str(e)

print(error_output)