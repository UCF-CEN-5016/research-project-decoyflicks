import os
import keras
import keras_nlp

# Set mixed precision policy (causes the error)
keras.mixed_precision.set_global_policy("mixed_float16")

# Switch backend if needed (error occurs with both)
# os.environ["KERAS_BACKEND"] = "tensorflow"  # or "jax"

# Attempt to load pretrained model
try:
    classifier = keras_nlp.models.BertClassifier.from_preset(
        "bert_tiny_en_uncased_sst2"
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

# Works when mixed precision is disabled
keras.mixed_precision.set_global_policy("float32")
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_tiny_en_uncased_sst2"
)
print("Model loaded successfully with float32 policy")