import os
import tensorflow as tf
from keras_nlp.models import BertClassifier

def load_model_with_precision(precision):
    try:
        tf.keras.mixed_precision.set_global_policy(precision)
        classifier = BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")

# Load model with mixed precision
load_model_with_precision("mixed_float16")

# Load model with float32 precision
load_model_with_precision("float32")