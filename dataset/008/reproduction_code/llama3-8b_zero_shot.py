import tensorflow as tf
from object_detection import model_holders

@tf.function
def load_model():
    return tf.saved_model.load('path_to_pretrained_model')

load_model()