import tensorflow as tf

print("Tensorflow version:", tf.__version__)

# Try to load the Faster RCNN model checkpoint (TF1 checkpoint) using TF2 code

# This URL points to the TF2 saved model, but let's try simulating the error by loading a TF1 checkpoint
# For demonstration, let's try to load a TF1 checkpoint directly which triggers error.

import os
import tempfile
import urllib.request
import tarfile

# Download the model checkpoint (TF1 checkpoint)
MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"

# Download and extract
model_dir = tempfile.mkdtemp()
archive_path = os.path.join(model_dir, "model.tar.gz")

urllib.request.urlretrieve(MODEL_URL, archive_path)

with tarfile.open(archive_path) as tar:
    tar.extractall(path=model_dir)

# The extracted folder name (usually)
extracted_folder = os.path.join(model_dir, "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8")

# Check for checkpoint files
print("Checkpoint files:", os.listdir(extracted_folder))

# Try to load checkpoint using tf.train.Checkpoint (TF2 style)
ckpt = tf.train.Checkpoint()

try:
    ckpt.restore(tf.train.latest_checkpoint(extracted_folder)).assert_existing_objects_matched()
    print("Checkpoint loaded successfully")
except Exception as e:
    print("Error loading checkpoint:", e)

# Alternatively, trying to load a TF1 graph with TF2 eager enabled triggers errors like "ValueError: not supported for tf2"