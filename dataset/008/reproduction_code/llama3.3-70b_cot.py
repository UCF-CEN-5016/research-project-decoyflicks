# Import necessary libraries
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Set up minimal environment
tf.version.VERSION  # Ensure TensorFlow version is 2.13.0
print(f"TensorFlow version: {tf.__version__}")

# Define the model configuration
model_config = """
model {
  faster_rcnn {
    num_classes: 90
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: 'faster_rcnn_inception_resnet_v2'
    }
  }
}
"""

# Load model configuration
model_config = config_util.get_configs_from_pipeline_file("path/to/model_config.pbtxt")
model_config = config_util.merge_external_params_with_configs(model_config, {"model": model_config})

# Attempt to build the model
model = model_builder.build(model_config, is_training=True)

# This should trigger the Value error if the TF2 version is incompatible
try:
    model_fn = model.create_model_fn()
    model_fn()
except ValueError as e:
    print(f"Error: {e}")