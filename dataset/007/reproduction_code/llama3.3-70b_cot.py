import tensorflow as tf
from object_detection.utils import config_util

# Set up minimal environment
pipeline_config_path = 'path_to_your_config_file.config'

# Create a file with invalid UTF-8 bytes
with open(pipeline_config_path, 'wb') as f:
    f.write(b'\xe9')  # Write a single byte with value 0xe9

# Try to read the file using the utf-8 codec
try:
    with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
        proto_str = f.read()
except UnicodeDecodeError as e:
    print(f"Error decoding file: {e}")

# To fix the issue, we can specify the encoding when opening the file
try:
    with tf.io.gfile.GFile(pipeline_config_path, 'rb') as f:
        proto_str = f.read().decode('latin1')  # Use latin1 encoding to decode the file
    print("File decoded successfully")
except UnicodeDecodeError as e:
    print(f"Error decoding file: {e}")