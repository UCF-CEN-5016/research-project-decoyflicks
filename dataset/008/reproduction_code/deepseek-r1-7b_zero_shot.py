import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

# Path to your pretrained Faster R-CNN model
model_path = 'path/to/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.h5'

# Load the configuration file
config = config_util.get configurable_file(model_path)
params = config['model'].get('params', {})

# Build the model
model = model_builder.build(params)

# Restore the model weights
 restoring_ops = model_builder.get_restoring_ops(model, params)
tf.train.init_from_checkpoint(params['init_checkpoint'], restore_ops)

# Now you can use the model with TF 2.x APIs
image_input = tf.keras.Input(shape=(640, 640, 3))
...