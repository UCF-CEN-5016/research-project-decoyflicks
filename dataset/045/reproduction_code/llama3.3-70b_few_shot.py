import os
import yaml

# Load config YAML
config_path = 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Simulate speech-to-speech generation
def generate_speech(config):
    # Check if key exists
    if 'input_feat_per_channel' not in config:
        raise KeyError("Key 'input_feat_per_channel' is not in struct")

    # Continue with generation
    print("Generating speech...")

# Load config and try to generate speech
try:
    generate_speech(config)
except KeyError as e:
    print(f"Error: {e}")

# To fix, add the key to the config YAML:
# input_feat_per_channel: <value>