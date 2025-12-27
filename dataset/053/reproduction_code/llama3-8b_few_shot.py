import os
import keras_nlp as knp

# Set Keras backend to TensorFlow
os.environ["Keras_Backend"] = 'tensorflow'

# Load the BertClassifier from preset
try:
    classifier = knp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
except Exception as e:
    print(f"Error: {e}")

print("Try to load weights...")
classifier.predict(["I love modular workflows in keras-nlp"])