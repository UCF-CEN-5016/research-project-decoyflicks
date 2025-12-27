import yaml

# Simulate loading the config.yaml file
config_yaml = """
# Example content of config.yaml but missing 'input_feat_per_channel'
model:
  input_feat_dim: 80
  encoder_layers: 12
  decoder_layers: 6
"""

def main():
    # Load config
    config = yaml.safe_load(config_yaml)
    
    # Here the code expects input_feat_per_channel inside model config
    # Let's access it directly to simulate the error
    try:
        input_feat_per_channel = config['model']['input_feat_per_channel']
        print(f"input_feat_per_channel: {input_feat_per_channel}")
    except KeyError as e:
        print(f"KeyError: {e} - This key is missing in config")

if __name__ == "__main__":
    main()