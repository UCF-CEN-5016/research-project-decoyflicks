import tensorflow as tf
from object_detection import model_main_tf2
from object_detection import exporter_main_v2

# Set up minimal environment
tf_version = tf.__version__
print(f"TensorFlow version: {tf_version}")

# Define the model and training parameters
model_name = "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8"
pipeline_config_path = f"models/{model_name}/pipeline.config"
model_dir = f"models/{model_name}/train"

# Train the model using model_main_tf2.py
# Note: This step is not actually executed in this code snippet, but it's a necessary step to reproduce the bug
# model_main_tf2.main([
#     "--model_dir",
#     model_dir,
#     "--pipeline_config_path",
#     pipeline_config_path,
#     "--alsologtostderr"
# ])

# Export the trained model using exporter_main_v2.py
# This will trigger the AttributeError
exporter_main_v2.main([
    "--input_type",
    "image_tensor",
    "--pipeline_config_path",
    pipeline_config_path,
    "--trained_checkpoint_dir",
    model_dir,
    "--output_directory",
    "exported_model"
])