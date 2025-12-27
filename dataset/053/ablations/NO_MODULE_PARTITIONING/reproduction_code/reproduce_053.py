import os
import tensorflow as tf
import keras
import keras_nlp

os.environ["Keras_Backend"] = 'tensorflow'
tf.keras.mixed_precision.set_global_policy('mixed_float16')

classifier = None
try:
    classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
except Exception as e:
    print(e)  # Expecting AttributeError: 'LossScaleOptimizerV3' object has no attribute 'name'