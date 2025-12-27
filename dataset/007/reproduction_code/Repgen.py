import os
import sys

# Clone TensorFlow models repository into a relative path
os.system('git clone https://github.com/tensorflow/models.git models/research/object_detection')

# Install required packages
os.system('pip install tensorflow==2.10.1')
os.system('pip install tensorflow-addons==0.20.0')

# Define relative paths for configuration and model files
config_path = os.path.join('object_detection', 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config')

# Copy configuration and model files to the relative destination
os.system(f'copy {config_path} models/research/object_detection')

# Define relative path for config_util.py
config_util_path = os.path.join('object_detection', 'utils', 'config_util.py')

# Edit config_util.py with specified encoding
with open(config_util_path, 'r+', encoding='latin-1') as f:
    content = f.read()
    f.seek(0)
    f.write(content)
    f.truncate()

# Run model_main_tf2.py using relative paths for configuration and model directory
os.system('python3.9 models/research/object_detection/model_main_tf2.py --pipeline_config_path=models/research/object_detection/ssd_efficientdet_d0_512x512_coco17_tpu-8.config --model_dir=training --alsologtostderr')
