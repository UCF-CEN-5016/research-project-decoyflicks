import os

# Set the pipeline configuration path and model directory
pipeline_config_path = "ssd_efficientdet_d0_512x512_coco17_tpu-8.config"
model_dir = "training"

# Ensure the pipeline configuration file exists
if not os.path.exists(pipeline_config_path):
    raise FileNotFoundError(f"Pipeline configuration file not found: {pipeline_config_path}")

# Run the model_main_tf2 script with the specified parameters
os.system(f"python3.9 model_main_tf2.py --pipeline_config_path={pipeline_config_path} --model_dir={model_dir} --alsologtostderr")