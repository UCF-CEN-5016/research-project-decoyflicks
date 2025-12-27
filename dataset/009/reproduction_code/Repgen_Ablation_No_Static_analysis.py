import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from export_saved_model_tpu_lib import export

# Set up pipeline configuration
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile('path_to_pipeline.config', 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Define input placeholder
placeholder_tensor = tf.placeholder(tf.string, shape=[None], name='encoded_image_string_tensor')

# Export model
export(pipeline_config, placeholder_tensor, './exported_model')