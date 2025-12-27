import os
from delf.python import extract_features_pb2
from delf.python.datasets import dataset_pb2

# Set up paths
config_path = 'delf_config_example.pbtxt'
list_images_path = 'list_images.txt'
output_dir = 'data/oxford5k_features'

# Create sample config and list images files
with open(config_path, 'w') as f:
    pass  # Placeholder for actual configuration content

with open(list_images_path, 'w') as f:
    f.write('/path/to/image1.jpg\n/path/to/image2.jpg')

# Run the extract_features script
os.system(f'python3 extract_features.py --config_path {config_path} --list_images_path {list_images_path} --output_dir {output_dir}')