import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True
    )

    print(f"Loading model from {model_dir}")
    
    # Load the SavedModel
    detection_model = tf.saved_model.load(str(model_dir))
    
    # Ensure serving signature exists and use it
    if 'serving_default' in detection_model.signatures:
        print("Using default serving signature")
        detection_model = detection_model.signatures['serving_default']
    else:
        raise ValueError(f"Serving signature 'serving_default' not found in model {model_dir}.")
    
    return detection_model

# Example usage
model_name = 'ssd EfficientNet'
detection_model = load_model(model_name)