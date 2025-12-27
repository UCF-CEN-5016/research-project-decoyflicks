# Start with necessary imports and setup
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2

def create_pipeline_config():
    config = pipeline_pb2.TrainEvalPipelineConfig()
    
    # Add the required parameters to appropriate sub-messages
    config.eval_input_reader.max_num_boxes_in_each_image = 500
    config.eval_input_reader.num_max_evals_per_image = 500
    
    config.model.faster_rcnn.second_stage_post_processing.batch_non_max suppression.max_detections_per_class = 500
    config.model.faster_rcnn.second_stage_post_processing.batch_non_max suppression.max_total_detections = 500
    
    return config

def train_model(config):
    # Use the created pipeline config to initiate training
    pass

# Usage example:
pipeline_config = create_pipeline_config()
train_model(pipeline_config)