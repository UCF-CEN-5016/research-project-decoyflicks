from fairseq.models.speech_to_speech.s2ut_transformer import S2UTTransformerModel
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
import torch

# Define default configuration with required keys
default_config = {
    'encoder_embed_dim': 256,
    'decoder_embed_dim': 256,
    'input_feat_per_channel': 80  # Specify input feature dimensions
}

# Merge default config with user-defined config
def merge_configs(default_config, user_config):
    config = default_config.copy()
    config.update(user_config)
    return config

# Handle missing configuration key
def process_model(config):
    try:
        model = S2UTTransformerModel.build_model(config, SpeechToSpeechTask)
        return model
    except KeyError as e:
        print(f"Error: {e}\nMissing 'input_feat_per_channel' in config")
        return None

# Define user configuration
user_config = {
    'encoder_embed_dim': 256,
    'decoder_embed_dim': 256,
    # 'input_feat_per_channel' is missing
}

# Merge default and user configuration
config = merge_configs(default_config, user_config)

# Create model with merged config
model = process_model(config)

if model is not None:
    # Generate dummy input
    dummy_input = torch.randn(1, config['input_feat_per_channel'], 100)  # (batch, features, time)
    output = model(dummy_input)