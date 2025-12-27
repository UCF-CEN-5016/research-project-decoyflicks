import os
import numpy as np
import requests
import tensorflow as tf
import tarfile

# Step 1: Install compatible TensorFlow version
!pip install tensorflow==2.10.0

# Step 2: Download and extract the model
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"
model_path = "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"

response = requests.get(model_url)
with open(model_path, "wb") as f:
    f.write(response.content)

with tarfile.open(model_path, "r:gz") as tar:
    tar.extractall()

# Step 3: Load the model
tf.keras.backend.clear_session()
model = tf.saved_model.load("faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8")

# Step 4: Run inference (example)
# Dummy input (adjust based on your input shape)
input_shape = (640, 640, 3)
input_data = np.random.rand(1, *input_shape).astype(np.float32)

# Run inference
infer = model.signatures["serving_default"]
outputs = infer(tf.constant(input_data))

# Print output
print("Inference outputs:", outputs)