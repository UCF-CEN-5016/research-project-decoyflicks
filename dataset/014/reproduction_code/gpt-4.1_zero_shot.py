import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

configpath = "pipeline.config"

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(configpath, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.eval_input_reader.max_number_of_boxes = 500