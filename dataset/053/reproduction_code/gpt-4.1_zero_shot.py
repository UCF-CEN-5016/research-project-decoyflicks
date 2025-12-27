import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow.keras import mixed_precision
import keras_nlp

mixed_precision.set_global_policy("mixed_float16")

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")