import tensorflow as tf
import tarfile
import os

# Download and extract model
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"
model_dir = "faster_rcnn_model"
os.makedirs(model_dir, exist_ok=True)
tf.keras.utils.get_file(model_dir, model_url, extract=True)

# Load model
model = tf.keras.models.load_model(os.path.join(model_dir, "faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8"))

# Attempt inference
input_tensor = tf.keras.Input(shape=(640, 640, 3))
output = model(input_tensor)