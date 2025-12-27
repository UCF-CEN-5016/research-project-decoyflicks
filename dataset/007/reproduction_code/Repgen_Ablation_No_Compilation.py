import sys

# Ensure Python version compatibility
if sys.version_info < (3, 9):
    raise RuntimeError("Python 3.9 or higher is required")

# Import necessary libraries
import tensorflow as tf
from object_detection.utils import config_util

# Load the configuration file
pipeline_config_path = "ssd_efficientdet_d0_512x512_coco17_tpu-8.config"
config = config_util.get_configs_from_pipeline_file(pipeline_config_path)

# Attempt to decode the pipeline configuration
try:
    decoded_config = tf.compat.v1.train.ConfigProto().ParseFromString(config)
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
    sys.exit(1)

# Save the decoded configuration
with open("decoded_pipeline_config.pbtxt", "w") as f:
    f.write(str(decoded_config))

print("Pipeline configuration successfully decoded and saved")