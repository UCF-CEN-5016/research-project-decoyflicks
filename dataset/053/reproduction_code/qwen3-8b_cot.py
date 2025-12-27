import os
import tensorflow as tf
import keras_nlp
from tensorflow.keras import mixed_precision

# Set up mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Load the BertClassifier from a preset
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased_sst2"
)

# Attempt to predict (this will trigger the error during model loading)
classifier.predict(["I love modular workflows in keras-nlp"])

# Disable mixed precision to avoid optimizer wrapping
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

# Load the model without mixed precision
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased_sst2"
)