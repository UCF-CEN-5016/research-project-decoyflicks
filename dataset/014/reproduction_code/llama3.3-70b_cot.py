import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Set up minimal environment
configpath = "path_to_your_pipeline_config_file"  # Update this path

# Load the pipeline configuration
config = config_util.get_configs_from_pipeline_file(configpath)

# Create a TrainEvalPipelineConfig object
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

# Read the pipeline configuration file
with tf.io.gfile.GFile(configpath, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Triggering condition: add max_number_of_boxes to eval_input_reader
try:
    pipeline_config.eval_input_reader.max_number_of_boxes = 500
except AttributeError as e:
    print(f"Error: {e}")

# Print the updated pipeline configuration
print(pipeline_config)

# To fix the issue, manually update the pipeline.config file with the required lines:
# pipeline_config.train_config.max_number_of_boxes = 500
# pipeline_config.eval_config.max_num_boxes_to_visualize = 500
# pipeline_config.eval_config.num_visualizations = 500
# pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = 500
# pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = 500
# pipeline_config.model.faster_rcnn.first_stage_max_proposals = 500