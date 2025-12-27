import tensorflow as tf
import pathlib

def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=base_url + model_file,
        untar=True
    )
    
    # Incorrectly using TF2's load_v2 API for TF1 model
    model_dir = pathlib.Path(model_dir) / "saved_model"
    print(f"Loading model from: {model_dir}")
    model = tf.saved_model.load_v2(str(model_dir))  # <-- TF2 API for TF1 model
    return model.signatures["serving_default"]

# Example usage that fails
try:
    detection_model = load_model("faster_rcnn_inception_v2")
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

# Dummy inference call (will fail due to empty tensors)
TEST_IMAGE_PATHS = ["test_image.jpg"]
for image_path in TEST_IMAGE_PATHS:
    print(f"Processing {image_path}")
    # This would fail with empty tensors and shape mismatches