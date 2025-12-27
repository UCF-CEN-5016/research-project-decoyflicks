import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf.text_format import Merge

# Define the path to the `pipeline.config` file
configpath = 'https://github.com/tensorflow/models/tree/master/research/object_detection'

# Load the `pipeline.config` file
configs = config_util.get_configs_from_pipeline_file(configpath)

# Create a new `TrainEvalPipelineConfig` object and merge the configuration from the file
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
Merge(configs['model'], pipeline_config)

try:
    # Attempt to set the value directly
    pipeline_config.eval_input_reader.max_number_of_boxes = 500
except AttributeError as e:
    print(e)  # Expected error message: 'AttributeError: 'google.protobuf.pyext._message.RepeatedCompositeContainer' object has no attribute 'max_number_of_boxes'"

# Remove the erroneous line and try accessing and setting the value through the repeated composite container
try:
    pipeline_config.eval_input_reader.max_number_of_boxes = 500
except AttributeError as e:
    print(e)  # Expected error message: 'AttributeError: 'google.protobuf.pyext._message.RepeatedCompositeContainer' object has no attribute 'max_number_of_boxes'"