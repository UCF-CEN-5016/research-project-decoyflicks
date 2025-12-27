import os
import tarfile
import requests
import tensorflow as tf

def fetch_model_archive(url: str, dest_path: str) -> None:
    """Download a model archive from a URL to a local file."""
    response = requests.get(url)
    with open(dest_path, "wb") as f:
        f.write(response.content)

def extract_archive(archive_path: str, target_dir: str) -> None:
    """Extract a tar archive into a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)

def load_saved_model(model_dir: str):
    """Load a TensorFlow SavedModel from the given directory."""
    model = tf.saved_model.load(model_dir)
    return model

def run_inference(model):
    """Run a single random inference on the provided model and return detections."""
    input_tensor = tf.random.normal([1, 640, 640, 3])
    detections = model(input_tensor)
    return detections

def main():
    model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"
    archive_path = "faster_rcnn.tar.gz"
    extract_dir = "faster_rcnn"

    fetch_model_archive(model_url, archive_path)
    extract_archive(archive_path, extract_dir)

    try:
        model = load_saved_model(extract_dir)
        detections = run_inference(model)
        print(detections)
    except ValueError as e:
        print("Model loading or inference failed:", e)

if __name__ == "__main__":
    main()