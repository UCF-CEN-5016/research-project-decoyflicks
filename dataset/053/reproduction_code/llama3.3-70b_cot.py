# Import necessary libraries
import os
import keras
from keras_nlp.models import BertClassifier
import tensorflow as tf

# Set up minimal environment
os.environ["Keras_Backend"] = 'tensorflow'

# Add triggering conditions
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Instantiate the BertClassifier from a preset
try:
    classifier = BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
    classifier.predict(["I love modular workflows in keras-nlp"])
except AttributeError as e:
    print(f"Error: {e}")