import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
keras.mixed_precision.set_global_policy("mixed_float16")
import keras_nlp

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")