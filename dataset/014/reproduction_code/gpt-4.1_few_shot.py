import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Path to a pipeline.config file (replace with an actual file path)
configpath = "path/to/pipeline.config"

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(configpath, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# This line triggers the AttributeError, because eval_input_reader is repeated (a list)
pipeline_config.eval_input_reader.max_number_of_boxes = 500