import os
import sys

# Setup virtual environment and install dependencies
os.system('conda create -n ML python=3.9')
os.system('conda activate ML')
os.system('pip install tensorflow==2.x tf-slim')

# Clone object_detection repository
os.system('git clone https://github.com/tensorflow/models.git /home/server/miniforge3/envs/ML/lib/python3.9/site-packages/object_detection')

# Navigate to research directory and copy model_main_tf2.py
os.chdir('/home/server/miniforge3/envs/ML/lib/python3.9/site-packages/object_detection/research')
os.system('cp object_detection/model_main_tf2.py /tmp')

# Create models directory and place centernet_mobilenetv2fp subdirectory
os.mkdir('/home/server/models')
os.mkdir('/home/server/models/centernet_mobilenetv2fp')
os.system('cp -r /home/server/miniforge3/envs/ML/lib/python3.9/site-packages/object_detection/research/object_detection/models/centernet_mobilenetv2fp/* /home/server/models/centernet_mobilenetv2fp')

# Copy exporter.py and comment out import statement
os.system('cp /home/server/miniforge3/envs/ML/lib/python3.9/site-packages/object_detection/research/object_detection/exporters/exporter.py /tmp')
with open('/tmp/exporter.py', 'r') as file:
    lines = file.readlines()
with open('/tmp/exporter.py', 'w') as file:
    for line in lines:
        if 'from tensorflow.contrib.quantize.python import graph_matcher' in line:
            continue
        file.write(line)

# Run model_main_tf2.py with existing exporter.py
os.system('python /tmp/model_main_tf2.py --model_dir=/home/server/models/centernet_mobilenetv2fp --pipeline_config_path=/home/server/models/centernet_mobilenetv2fp/pipeline.config')