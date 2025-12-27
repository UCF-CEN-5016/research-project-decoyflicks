import torch
from x_transformers import RotaryEmbedding, Attention, TransformerWrapper
import einops  # Importing einops to fix the undefined variable issue

torch.manual_seed(42)

batch_size = 1
sequence_length = 32
x = torch.randint(0, 1000, (batch_size, sequence_length))
pos = torch.randn(batch_size, sequence_length)

rotary_embedding = RotaryEmbedding(dim=64, use_xpos=True, scale_base=512, interpolation_factor=1.0)
attention_layer = Attention(dim=64, heads=8, causal=True, rotary_embed_values=True)
transformer_wrapper = TransformerWrapper(num_tokens=1000, max_seq_len=32, attn_layers=attention_layer)

try:
    output = transformer_wrapper(x, pos=pos)
except Exception as e:
    # Check if the exception is an EinopsError related to the specific pattern
    if isinstance(e, einops.EinopsError) and 'Error while processing rearrange-reduction pattern "n -> n 1"' in str(e):
        print(f"Exception caught: {e}")
        print(f"Input tensor shape: {e.input_shape}")  # This will help in reproducing the bug