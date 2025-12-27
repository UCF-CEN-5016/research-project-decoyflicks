import tensorflow as tf
from official.vision.utils.object_detection import visualization_utils

# Set CUDA version to 11.5 and TensorFlow version to 2.4
# Define Python version as 3.7.16

model_name = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
input_image = tf.random.uniform((1, 640, 640, 3))

def initialize_detection_module(model_name):
    return tf.keras.models.load_model(model_name)

detection_module = initialize_detection_module(model_name)

try:
    outputs = detection_module.outputs
except AttributeError as e:
    print(e)