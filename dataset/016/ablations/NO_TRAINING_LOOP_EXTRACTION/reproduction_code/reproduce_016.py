import os
import tensorflow as tf
from official.legacy.image_classification import classifier_trainer
from official.legacy.image_classification.configs.base_configs import DataConfig

# Set environment variables
os.environ['DATA_DIR'] = '/scratch/cdacapp1/sowmya/actual_imagenet/tf_records'
os.environ['MODEL_DIR'] = '/home/cdacapp1/scaling/models-master/official/legacy/image_classification/checkpointss'

# Activate conda environment
os.system('source /home/apps/anaconda3/bin/activate && conda activate tf11')

# Define worker hosts in configuration files
gpu_yaml = """
task_index: 0
num_gpus: 2
cluster:
  worker:
    - <IP_ADDRESS_NODE_1>:<PORT>
    - <IP_ADDRESS_NODE_2>:<PORT>
"""
with open('gpu.yaml', 'w') as f:
    f.write(gpu_yaml)

gpu1_yaml = """
task_index: 1
num_gpus: 2
cluster:
  worker:
    - <IP_ADDRESS_NODE_1>:<PORT>
    - <IP_ADDRESS_NODE_2>:<PORT>
"""
with open('gpu1.yaml', 'w') as f:
    f.write(gpu1_yaml)

# Run training script on both nodes
os.system('python3 classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --model_dir=$MODEL_DIR --data_dir=$DATA_DIR --config_file=configs/examples/resnet/imagenet/gpu.yaml &')
os.system('python3 classifier_trainer.py --mode=train_and_eval --model_type=resnet --dataset=imagenet --model_dir=$MODEL_DIR --data_dir=$DATA_DIR --config_file=configs/examples/resnet/imagenet/gpu1.yaml &')

# Monitor logs and GPU usage
os.system('nvidia-smi')