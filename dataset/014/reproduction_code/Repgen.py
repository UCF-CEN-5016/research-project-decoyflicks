import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils.config_util import get_configs_from_pipeline_file

configpath = 'path/to/pipeline.config'

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with open(configpath, 'r') as f:
    proto_str = f.read()
text_format.Merge(proto_str, pipeline_config)

# Add the line `pipeline_config.eval_input_reader.max_number_of_boxes = 500` directly to the pipeline_config
pipeline_config.eval_input_reader.max_number_of_boxes = 500

# Save the modified pipeline_config back to the original file path (configpath) or a new file path
with open(configpath, 'w') as f:
    text_format.WriteToString(pipeline_config, f)

# Load the modified pipeline.config file again using config_util.get_configs_from_pipeline_file(configpath)
pipeline_config = get_configs_from_pipeline_file(configpath)

# Extract the eval_input_reader from the loaded pipeline_config and assign it to a variable named eval_input_reader
eval_input_reader = pipeline_config.eval_input_reader

# Call `eval_input_reader.max_number_of_boxes` to attempt to retrieve its value and capture any potential error or output
max_boxes = eval_input_reader.max_number_of_boxes