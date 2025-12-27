import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Path to pipeline.config file
configpath = 'path/to/pipeline.config'

# Load pipeline config from file
config = config_util.get_configs_from_pipeline_file(configpath)

# Create a TrainEvalPipelineConfig instance
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

# Read the pipeline config file as a proto string
with tf.io.gfile.GFile(configpath, "r") as f:
    proto_str = f.read()
text_format.Merge(proto_str, pipeline_config)

# Set max_number_of_boxes to 500 (triggering condition)
pipeline_config.eval_input_reader.max_number_of_boxes = 500

print("Reproducing the bug...")