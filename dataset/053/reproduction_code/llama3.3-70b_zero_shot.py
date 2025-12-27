import os
import tensorflow as tf
from keras_nlp.models import BertClassifier

# Setup mixed precision policy
mixed_precision = tf.keras.mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Create BertClassifier instance
classifier = BertClassifier.from_preset("bert_tiny_en_uncased_sst2")