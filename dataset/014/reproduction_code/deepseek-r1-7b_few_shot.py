import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

# Simulate creating a pipeline config file content string
pipeline_config_content = """
    model: "faster_rcnn"
    training: {
        ...other_training_params...
        eval_input_reader: {
            max_number_of_boxes: 500
            ...
        }
        ...
    }
"""
config_path = tf.io.gfile temps.open(tempfile.mktemp()) 

# Write the content to a temporary file
with open(config_path, 'w') as f:
    f.write(pipeline_config_content)

try:
    config = config_util.get_configs_from_pipeline_file(config_path)
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with tf.io.gfile.GFile(config_path, "r") as f: 
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)  
    
    # Add the problematic line
    pipeline_config.eval_input_reader.max_number_of_boxes = 500
except AttributeError as e:
    print(f"Error adding max_number_of_boxes: {e}")