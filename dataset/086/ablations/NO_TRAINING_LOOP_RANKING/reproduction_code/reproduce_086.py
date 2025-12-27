import torch
from einops import rearrange
from x_transformers import Encoder
import einops  # Importing einops to avoid undefined variable error

torch.manual_seed(42)

batch_size = 1
input_seq_length = 64
context_seq_length = 128

# Generate random input tensor and masks
x = torch.randn((batch_size, input_seq_length, 256))
mask = torch.ones((batch_size, input_seq_length)).bool()
context = torch.randn((batch_size, context_seq_length, 512))
context_mask = torch.ones((batch_size, context_seq_length)).bool()

# Initialize the Encoder model with rotary position embedding enabled
model = Encoder(
    dim=256,
    depth=4,
    heads=4,
    rotary_pos_emb=True,
    cross_attend=True,
    cross_attn_dim_context=512
)

# Create position tensors for input and context
pos = torch.arange(input_seq_length)
context_pos = torch.arange(context_seq_length)

# Attempt to run the model and catch specific errors
try:
    embed = model(
        x=x,
        mask=mask,
        context=context,
        pos=pos,
        context_pos=context_pos,
        context_mask=context_mask
    )
except Exception as e:
    # Check for specific einops error related to tensor shape
    if isinstance(e, einops.EinopsError):
        assert 'Error while processing rearrange-reduction pattern "n -> n 1"' in str(e)
        assert 'Input tensor shape: torch.Size([1, 32])' in str(e)