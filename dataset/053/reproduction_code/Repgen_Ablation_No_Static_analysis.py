import os
import keras_nlp
import tensorflow as tf

tf.config.experimental.set_policy(tf.keras.mixed_precision.Policy('mixed_float16'))

classifier = keras_nlp.models.BertClassifier.from_preset('bert_tiny_en_uncased_sst2')
test_inputs = ['I love modular workflows in keras-nlp']
predictions = classifier.predict(test_inputs)

assert predictions is not None and predictions.shape == (1, 2)