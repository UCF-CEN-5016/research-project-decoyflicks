import os
from google.protobuf import text_format
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2

# Set the path to a pipeline.config file that does not contain the lines causing the bug
configpath = 'path/to/your/pipeline.config'

# Load the existing pipeline.config file
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(configpath, 'r') as f:
    proto_str = f.read()
text_format.Merge(proto_str, pipeline_config)

try:
    # Attempt to add the line directly
    pipeline_config.eval_input_reader.max_number_of_boxes = 500
except AttributeError as e:
    print(e)  # Should raise an error

# Open a second pipeline.config file and manually edit it to include all the lines causing the bug
configpath_edited = 'path/to/your/editied_pipeline.config'

# Load the edited pipeline.config file
pipeline_config_edited = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(configpath_edited, 'r') as f:
    proto_str = f.read()
text_format.Merge(proto_str, pipeline_config_edited)

# Attempt to add the line directly after editing
pipeline_config_edited.eval_input_reader.max_number_of_boxes = 500

# Compare the performance of models trained with both configurations on images with over 100 objects per image