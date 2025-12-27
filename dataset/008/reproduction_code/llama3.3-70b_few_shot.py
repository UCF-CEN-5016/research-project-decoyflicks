import tensorflow as tf

# Load pre-trained model
model_url = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz"

# Try to load the model (this will cause the error)
try:
    model = tf.saved_model.load(model_url)
    print("Model loaded successfully")
except ValueError as e:
    print(f"Error loading model: {e}")

# The error occurs because the pre-trained model is not compatible with TensorFlow 2.13.0
# To fix this, you would need to either use an earlier version of TensorFlow or update the model