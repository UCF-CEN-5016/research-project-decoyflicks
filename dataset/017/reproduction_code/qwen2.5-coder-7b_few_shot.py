import pathlib
import tensorflow as tf

def _download_saved_model_dir(model_name):
    base_url = 'http://download.tensorflow.org/models/research/object_detection/'
    archive_name = model_name + '.tar.gz'
    extracted_path = tf.keras.utils.get_file(
        fname=archive_name,
        origin=base_url + archive_name,
        untar=True
    )
    return pathlib.Path(extracted_path) / "saved_model"

def load_model(model_name):
    saved_model_dir = _download_saved_model_dir(model_name)
    print(f"Loading model from: {saved_model_dir}")
    loaded = tf.saved_model.load(str(saved_model_dir))
    return loaded.signatures["serving_default"]

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