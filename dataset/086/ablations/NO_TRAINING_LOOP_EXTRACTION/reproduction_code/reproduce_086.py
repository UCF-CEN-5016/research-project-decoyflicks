import torch
from einops import rearrange
from x_transformers import XTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
sequence_length = 32

input_tensor = torch.rand(batch_size, sequence_length).to(device)
config = {'rotary_xpos': True}

model = XTransformer(rotary_xpos=config['rotary_xpos']).to(device)

try:
    output = model(input_tensor)
except Exception as e:
    if isinstance(e, rearrange.EinopsError):
        print(f"Caught an EinopsError: {e}")
        print(f"Input tensor shape: {input_tensor.shape}")