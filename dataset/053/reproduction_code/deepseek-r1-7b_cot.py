import tensorflow as tf
from keras_nlp import models, task

# Load the preset model without mixed precision to avoid the error
bert = models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
bert.predict(["I love modular workflows in keras-nlp"])

# Uncomment this line if needed
tf.keras.mixed_precision.set_global_policy('mixed_float16')

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")