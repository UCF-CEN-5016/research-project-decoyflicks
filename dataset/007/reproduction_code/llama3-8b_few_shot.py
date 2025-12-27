import tensorflow as tf

# Load pipeline configuration file (similar to your issue)
config_file = 'path/to/your/pipeline_config.pbtxt'

with open(config_file, 'rb') as f:
    proto_str = f.read()

# Attempt to parse the protocol buffer string
configs = tf.config.parse_proto(proto_str)

print("Configs:", configs)  # Should fail with UnicodeDecodeError