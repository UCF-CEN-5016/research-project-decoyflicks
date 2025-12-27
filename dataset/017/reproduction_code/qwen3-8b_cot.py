import os
import tarfile
import requests
import pathlib
import tensorflow as tf
import numpy as np
from PIL import Image

# Set up paths
model_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu_8'
base_url = 'http://download.tensorflow.org/models/research/object_detection/'
model_file = model_name + '.tar.gz'
model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True
)

model_dir = pathlib.Path(model_dir) / "saved_model"
print("Model directory:", model_dir)

# Attempt to load the model using TF2 API
try:
    model = tf.saved_model.load_v2(str(model_dir))
    print("Model loaded successfully (TF2 API).")
except Exception as e:
    print("Failed to load model with TF2 API:", e)

# Function to run inference
def show_inference(model, image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize((320, 320))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    input_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    detections = model.signatures['serving_default'](input_tensor)
    print("Detections:", detections)

# Test inference
TEST_IMAGE_PATHS = ["test_image.jpg"]  # Replace with actual image path
for image_path in TEST_IMAGE_PATHS:
    print(f"Running inference on {image_path}")
    show_inference(model, image_path)