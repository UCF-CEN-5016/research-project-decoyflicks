import subprocess
import os
import tensorflow as tf

# Set TensorFlow version
os.environ['TF_VERSION'] = 'tf-nightly-cpu==2.16.0.dev20231103'
os.environ['TF_KERAS_VERSION'] = 'tf-keras==2.16.0'

# Define parameters
batch_size = 32
input_image_size = [224, 224]
data_dir = '/dir_to_imagenet_data'

# Command to run the classifier_trainer.py script
command = [
    'python3', 'classifier_trainer.py',
    '--mode=train_and_eval',
    '--model_type=resnet',
    '--dataset=imagenet',
    f'--data_dir={data_dir}'
]

# Execute the command
try:
    output = subprocess.check_output(command, stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    output = e.output

# Check for the specific AttributeError
if b"AttributeError: module 'tf_keras.optimizers.legacy' has no attribute 'Optimizer'. Did you mean: 'optimizer_v2'?" in output:
    print("Bug reproduced: AttributeError found.")
else:
    print("No AttributeError found.")