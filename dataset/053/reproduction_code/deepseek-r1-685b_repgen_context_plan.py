import os
import keras
import keras_nlp

def load_pretrained_model():
    try:
        # Set mixed precision policy
        keras.mixed_precision.set_global_policy("mixed_float16")
        classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")

def load_model_with_float32_policy():
    # Set float32 precision policy
    keras.mixed_precision.set_global_policy("float32")
    classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")
    print("Model loaded successfully with float32 policy")

# Main code
load_pretrained_model()
load_model_with_float32_policy()