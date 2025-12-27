import os
import keras_nlp as knp

# Set the Keras backend to TensorFlow
os.environ["Keras_Backend"] = 'tensorflow'

try:
    # Try to instantiate BertClassifier from preset
    classifier = knp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
except AttributeError as e:
    print(f"Caught exception: {e}")

# Comment out the line setting mixed precision policy (optional)
# os.environ["mixed_precision_policy"] = "mixed_float16"