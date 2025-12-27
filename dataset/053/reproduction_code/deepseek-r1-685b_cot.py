import os
import keras
import keras_nlp

# Set mixed precision policy - this triggers the bug
keras.mixed_precision.set_global_policy("mixed_float16")

# Try to load a preset model (fails)
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")

classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
keras.mixed_precision.set_global_policy("mixed_float16")  # After model creation