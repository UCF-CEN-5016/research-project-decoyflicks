from fairseq.models.speech_to_speech.s2ut_transformer import S2UTTransformerModel
from fairseq.tasks.speech_to_speech import SpeechToSpeechTask
import torch

# Define model configuration with required keys
config = {
    'encoder_embed_dim': 256,
    'decoder_embed_dim': 256,
    'input_feat_per_channel': 80  # Specify input feature dimensions
}

try:
    # Create model with complete config
    model = S2UTTransformerModel.build_model(config, SpeechToSpeechTask)
    
    # Generate dummy input
    dummy_input = torch.randn(1, 80, 100)  # (batch, features, time)
    
    # Process input through the model
    output = model(dummy_input)
    
    print("Model processing successful.")
    
except KeyError as e:
    print(f"Error: {e}\nMissing required key in config: 'input_feat_per_channel'")
except Exception as ex:
    print(f"An error occurred: {ex}")