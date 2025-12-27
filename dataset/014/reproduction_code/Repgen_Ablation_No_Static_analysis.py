import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder_tf2

# Set the path to the pipeline.config file
configpath = "path_to_pipeline_config_file"

# Load the configuration settings into a dictionary
pipeline_config_dict = config_util.get_configs_from_pipeline_file(configpath)

# Read the contents of the pipeline.config file and parse it
with tf.io.gfile.GFile(configpath, 'r') as f:
    text_format.Merge(f.read(), pipeline_config_dict['model'])

# Attempt to set max_number_of_boxes
pipeline_config_dict['eval_input_reader']['max_number_of_boxes'] = 500

# Modify the pipeline.config file directly
with open(configpath, "a") as f:
    f.write("\npipeline_config.train_config.max_number_of_boxes = 500\n")
    f.write("pipeline_config.eval_config.max_num_boxes_to_visualize = 500\n")
    f.write("pipeline_config.eval_config.num_visualizations = 500\n")
    f.write("pipeline_config.eval_input_reader.max_number_of_boxes = 500\n")
    f.write("pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = 500\n")
    f.write("pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = 500\n")
    f.write("pipeline_config.model.faster_rcnn.first_stage_max_proposals = 500\n")

# Re-read the modified pipeline.config file and parse it
with tf.io.gfile.GFile(configpath, 'r') as f:
    text_format.Merge(f.read(), pipeline_config_dict['model'])

# Set max_number_of_boxes again
pipeline_config_dict['eval_input_reader']['max_number_of_boxes'] = 500

# Load preprocessed images for evaluation
dataset = tf.data.Dataset.list_files("path_to_preprocessed_images/*.tfrecord")
decoder = tf.compat.v1.image.decode_image(tf.io.read_file(dataset))
dataset = dataset.map(lambda x: decoder(x))

# Initialize an evaluator object
evaluator = model_builder_tf2.create_evaluator(pipeline_config_dict['eval_config'])

# Run the evaluator on the preprocessed dataset
evaluation_results = evaluator.evaluate_next_step(dataset, num_steps=10)

# Verify evaluation results
print(evaluation_results)