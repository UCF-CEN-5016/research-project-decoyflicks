import tensorflow as tf
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder_tf2 as model_builder
from object_detection.utils import config_util

# Define batch size and image dimensions
batch_size = 1
height, width = 640, 640

# Create random uniform input data
inputs = tf.random.uniform((batch_size, height, width, 3), dtype=tf.float32)

# Parse the pipeline configuration file
pipeline_config_path = 'path/to/pipeline_config_file.config'
configs = config_util.get_configs_from_pipeline_proto_string(tf.io.read_file(pipeline_config_path))
pipeline_config = configs['model']

# Build the graph for Faster-RCNN detection
detection_model = model_builder.build(model_config=model_config, is_training=False)
result_tensor_dict = detection_model(inputs)

# Initialize TensorFlow session and restore variables from checkpoint
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list=tf.global_variables())
saver.restore(sess, 'path/to/ckpt_path')

# Verify that the placeholder_tensor and result_tensor_dict are correctly defined without errors

# Export the model as SavedModel
tf.saved_model.save(detection_model, 'path/to/export_dir', inputs={'input': detection_model.input})

# Monitor GPU memory usage during the export process

# Assert that the exported SavedModel does not contain an attribute named 'outputs'

# Load the saved model from directory
with tf.saved_model.load('path/to/export_dir') as loaded_model:
    run_inference_from_saved_model(
        inputs,
        saved_model_dir='path/to/export_dir',
        repeat=1
    )

# Verify that calling run_inference_from_saved_model raises an AttributeError indicating 'DetectionFromImageModule' object has no attribute 'outputs'