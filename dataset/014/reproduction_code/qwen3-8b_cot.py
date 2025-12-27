import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Set the path to your pipeline.config file
configpath = "path/to/your/pipeline.config"

# Load the existing pipeline config
config = config_util.get_configs_from_pipeline_file(configpath)

# Initialize the pipeline config object
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(configpath, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Attempt to set a non-existent field in the proto
# This will raise AttributeError: 'RepeatedCompositeContainer' object has no attribute 'max_number_of_boxes'
pipeline_config.eval_input_reader.max_number_of_boxes = 500

# Correct way to set max_number_of_boxes in the model
pipeline_config.model.faster_rcnn.first_stage_max_proposals = 500
pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = 500
pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = 500