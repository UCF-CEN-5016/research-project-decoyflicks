import os
import tarfile
import requests
import pathlib
import tensorflow as tf
import numpy as np
from PIL import Image

def download_and_extract_model(model_name, base_url):
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True
    )
    return pathlib.Path(model_dir) / "saved_model"

def load_model(model_dir):
    try:
        model = tf.saved_model.load(str(model_dir))
        print("Model loaded successfully (TF2 API).")
        return model
    except Exception as e:
        print("Failed to load model with TF2 API:", e)
        return None

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((320, 320))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def run_inference(model, image_path):
    input_tensor = tf.convert_to_tensor(preprocess_image(image_path), dtype=tf.float32)
    detections = model.signatures['serving_default'](input_tensor)
    print("Detections:", detections)

# Set up paths
model_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu_8'
base_url = 'http://download.tensorflow.org/models/research/object_detection/'

model_dir = download_and_extract_model(model_name, base_url)
print("Model directory:", model_dir)

# Load the model
model = load_model(model_dir)

# Test inference
TEST_IMAGE_PATHS = ["test_image.jpg"]  # Replace with actual image path
for image_path in TEST_IMAGE_PATHS:
    print(f"Running inference on {image_path}")
    run_inference(model, image_path)