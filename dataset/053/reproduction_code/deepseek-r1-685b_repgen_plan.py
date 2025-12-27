import os
import keras
import keras_nlp

def load_model_with_precision(precision):
    try:
        # Set mixed precision policy
        keras.mixed_precision.set_global_policy(precision)

        # Attempt to load pretrained model
        classifier = keras_nlp.models.BertClassifier.from_preset(
            "bert_tiny_en_uncased_sst2"
        )
        print(f"Model loaded successfully with {precision} policy")
    except Exception as e:
        print(f"Error occurred with {precision} policy: {type(e).__name__}: {e}")

# Load model with mixed_float16 policy
load_model_with_precision("mixed_float16")

# Load model with float32 policy
load_model_with_precision("float32")