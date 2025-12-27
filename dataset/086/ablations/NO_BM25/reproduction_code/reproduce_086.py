import torch
from einops import rearrange, EinopsError  # Import EinopsError to handle specific exceptions
from x_transformers import AdaptiveRMSNorm

torch.manual_seed(42)

batch_size = 1
dim = 32
x = torch.rand(batch_size, 1, dim)  # Input tensor with shape [1, 1, 32]
condition = torch.rand(batch_size, dim)  # Condition tensor with shape [1, 32]

adaptive_rms_norm = AdaptiveRMSNorm(dim=dim)

try:
    output = adaptive_rms_norm(x, condition=condition)
    rotary_xpos = True
    power = torch.tensor([0])  # Power tensor for scaling
    scale = adaptive_rms_norm.scale ** rearrange(power, 'n -> n 1')  # This line may cause the bug
except Exception as e:
    if isinstance(e, EinopsError):  # Catching the specific EinopsError
        print(f"Error: {e}")
        print(f"Input tensor shape: {x.shape}")  # Print the shape of the input tensor for debugging