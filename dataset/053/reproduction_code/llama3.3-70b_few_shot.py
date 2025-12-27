import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

# Set mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# Load BertClassifier with mixed precision
try:
    classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
except AttributeError as e:
    print(f"Error: {e}")

# To fix the issue, comment out the mixed precision policy line
# mixed_precision.set_global_policy('mixed_float16')