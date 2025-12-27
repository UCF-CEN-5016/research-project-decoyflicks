import tensorflow as tf

# Ensure TensorFlow version is 2.x
assert tf.__version__.startswith('2.')

# Import necessary modules from object_detection library
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Define the path to the pipeline configuration file
pipeline_config_path = 'models/centernet_mobilenetv2fp/pipeline.config'

# Load the pipeline configuration file
with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
    config_str = f.read()

# Parse the pipeline configuration file into a message object
config = pipeline_pb2.TrainEvalPipelineConfig()
text_format.Merge(config_str, config)

# Build the model from the parsed configuration
model_fn = model_builder.build(
    model_config=config.model,
    is_training=True)