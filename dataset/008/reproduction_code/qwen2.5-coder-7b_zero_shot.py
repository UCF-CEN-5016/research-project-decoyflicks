import tensorflow as tf
from object_detection import model_holders

MODEL_PATH = 'path_to_pretrained_model'

@tf.function
def load_pretrained_model() -> tf.Module:
    """Load and return a TensorFlow SavedModel from MODEL_PATH."""
    return tf.saved_model.load(MODEL_PATH)

# Preserve original behavior: load model at import time
load_pretrained_model()