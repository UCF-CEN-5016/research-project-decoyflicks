import tensorflow as tf

# Import necessary modules and functions from object_detection package
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Define the model directory and pipeline configuration path
model_dir = 'models/centernet_mobilenetv2fp'
pipeline_config_path = 'models/centernet_mobilenetv2fp/pipeline.config'

# Load the pipeline configuration
config = config_util.get_configs_from_pipeline_file(pipeline_config_path)

# Build the model using the loaded configuration
model_fn = model_builder.build(
    model_config=config['model'],
    is_training=True
)