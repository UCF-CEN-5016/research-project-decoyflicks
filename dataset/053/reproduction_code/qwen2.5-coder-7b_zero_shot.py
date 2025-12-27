import os
from keras_nlp import models
os.environ["Keras_Backend"] = 'tensorflow'

try:
    classifier = models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
except Exception as e
    print(f"Error: {e}")