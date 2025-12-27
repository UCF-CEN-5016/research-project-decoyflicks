import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

configpath = 'path/to/your/configfile.pbtxt'

def fix_pipeline_config(configpath):
    """Modifies the config to resolve the protobuf field access issue."""
    with tf.io.gfile.GFile(configpath, "r") as f:
        proto_str = f.read()
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        text_format.Merge(proto_str, pipeline_config)
        
        # Access the max_number_of_boxes in eval_input_reader via its container
        evaluation_input_config = pipeline_config.eval_input_reader
        boxes_container = evaluation_input_config.max_number_of_boxes
        boxes_container[0] = 500  # Adjust index as needed
        
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        text_format.Merge(proto_str, pipeline_config)
        
        # Optionally also set other required fields similar to the above approach

    return pipeline_config

# Usage example
config_util.get_configs_from_pipeline_file(configpath).replace(
    'path/to/your/configfile.pbtxt',
    fix_pipeline_config(configpath)
)

# Note: The function returns a new ConfigWrapper with updated values.