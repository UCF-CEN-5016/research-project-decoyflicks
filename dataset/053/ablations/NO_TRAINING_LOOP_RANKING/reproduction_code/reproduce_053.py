import os
import keras
import keras_nlp
from keras import mixed_precision

os.environ["Keras_Backend"] = 'tensorflow'
mixed_precision.set_global_policy('mixed_float16')

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")