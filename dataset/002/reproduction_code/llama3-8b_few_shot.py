import tensorflow as tf

# Minimal setup
model_dir = 'models/centernet_mobilenetv2fp'
pipeline_config_path = f'{model_dir}/pipeline.config'

try:
    # Trigger the error
    from tensorflow.contrib.quantize.python import graph_matcher  # <--- Error here
except ModuleNotFoundError as e:
    print(f"Error: {e}")

print("This line should be executed")