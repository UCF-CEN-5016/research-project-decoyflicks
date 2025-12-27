import yaml

# Load and update the config file to include 'input_feat_per_channel'
config = yaml.safe_load(open('/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml'))
config['task'] = {'speech_to_speech': {'input_feat_per_channel': 128}}  # Example value
with open('/root/autodl-tmp/FormattingData/DATA_ROOT/config.yaml', 'w') as file:
    yaml.dump(config, file)

# Update config.yaml by adding or modifying 'input_feat_per_channel'
import yaml

# Load existing configuration
with open('config.yaml', 'r') as f:
    original_config = yaml.safe_load(f)

# Modify the configuration to include 'input_feat_per_channel' for speech_to_speech
original_config['task']['speech_to_speech'] = {
    'sample_rate': 16000,
    'n_src necessarily include 'input_feat_per_channel' in your config.yaml file. Ensure it's set appropriately based on your model's requirements.
}

# Save the updated configuration back to the file
with open('config.yaml', 'w') as f:
    yaml.dump(original_config, f)