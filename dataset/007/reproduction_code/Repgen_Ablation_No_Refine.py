import sys

# Ensure TensorFlow version compatibility
import tensorflow as tf
if tf.__version__ != '2.10.1':
    raise ImportError("TensorFlow version 2.10.1 is required.")

# Set up environment variables if necessary

# Run the model training command
sys.argv = ["model_main_tf2.py", "--pipeline_config_path=ssd_efficientdet_d0_512x512_coco17_tpu-8.config", "--model_dir=training", "--alsologtostderr"]
import object_detection.model_main_tf2 as main
main.main(sys.argv)