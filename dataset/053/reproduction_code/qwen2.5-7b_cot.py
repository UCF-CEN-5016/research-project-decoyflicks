import tensorflow as tf
import keras_nlp
from tensorflow.keras import mixed_precision

def load_bert_classifier(preset_name):
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    
    classifier = keras_nlp.models.BertClassifier.from_preset(preset_name)
    
    return classifier

def predict_with_classifier(classifier, input_text):
    predictions = classifier.predict([input_text])
    return predictions

def disable_mixed_precision():
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)

preset_name = "bert_tiny_en_uncased_sst2"

# Load the BertClassifier with mixed precision
classifier = load_bert_classifier(preset_name)

# Attempt to predict
predicted_output = predict_with_classifier(classifier, "I love modular workflows in keras-nlp")

# Disable mixed precision
disable_mixed_precision()

# Load the model without mixed precision
classifier = load_bert_classifier(preset_name)