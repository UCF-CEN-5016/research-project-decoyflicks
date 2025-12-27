import os
import tensorflow as tf
from keras_nlp.models import BertClassifier

# Set up mixed precision policy
os.environ["KERAS_BACKEND"] = "tensorflow"
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Attempt to load pretrained model with mixed precision
try:
    classifier = BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
    classifier.predict(["I love modular workflows in keras-nlp"])
except AttributeError as e:
    print(f"Error: {e}")