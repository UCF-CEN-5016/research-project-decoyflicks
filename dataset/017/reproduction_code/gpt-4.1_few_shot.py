import tensorflow as tf
import pathlib

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"
    print("Model directory:", model_dir)

    # Incorrect loading method for TF1 SavedModel
    model = tf.saved_model.load_v2(str(model_dir))  
    signature = model.signatures["serving_default"]
    print("Signature keys:", list(signature.structured_outputs.keys()))

    return model

# Example usage with a TF1 model name (e.g. ssd_mobilenet_v1_coco_2017_11_17)
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

# Attempt to run inference (this will fail because model loaded incorrectly)
# Dummy input for signature
import numpy as np
dummy_input = tf.constant(np.zeros((1, 300, 300, 3), dtype=np.uint8))

try:
    output = detection_model.signatures["serving_default"](dummy_input)
    print(output)
except Exception as e:
    print("Inference failed:", e)