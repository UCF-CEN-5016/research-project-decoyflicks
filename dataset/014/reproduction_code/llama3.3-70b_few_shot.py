import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Load pipeline configuration from file
config_path = 'path_to_pipeline_config_file'  # Update with actual file path
config = config_util.get_configs_from_pipeline_file(config_path)

# Create a new pipeline configuration
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Attempt to set max_number_of_boxes for eval_input_reader (this will throw an error)
try:
    pipeline_config.eval_input_reader.max_number_of_boxes = 500
except AttributeError as e:
    print(f"Error: {e}")

# Correct way to set max_number_of_boxes for eval_input_reader
# Update the pipeline.config file directly or use a different approach to configure the model