import os
import tensorflow as tf
from tensorflow import keras
import keras_nlp

# Set mixed precision policy to mixed_float16
# This triggers the AttributeError during model loading
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Attempt to load a pretrained Keras_NLP BertClassifier preset
# This line raises:
# AttributeError: 'LossScaleOptimizerV3' object has no attribute 'name'
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")

# If the above line succeeds, run a prediction
print(classifier.predict(["This is a test sentence."]))