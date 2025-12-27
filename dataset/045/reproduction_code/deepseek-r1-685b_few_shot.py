from fairseq.models.speech_to_speech.s2ut_transformer import S2UTTransformerModel
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
import torch

# Minimal config missing required key
config = {
    'encoder_embed_dim': 256,
    'decoder_embed_dim': 256,
    # Missing 'input_feat_per_channel' which should specify input feature dimensions
}

# Create model with incomplete config
model = S2UTTransformerModel.build_model(config, SpeechToSpeechTask)

# This will raise the KeyError when trying to process input
dummy_input = torch.randn(1, 80, 100)  # (batch, features, time)
try:
    output = model(dummy_input)
except KeyError as e:
    print(f"Error: {e}\nMissing 'input_feat_per_channel' in config")