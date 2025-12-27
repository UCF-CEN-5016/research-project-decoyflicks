import torch
from model import ContinuousTransformerWrapper, Decoder  # Assuming these are defined elsewhere

# Define the model with the problematic configuration
net = ContinuousTransformerWrapper(
    dim_out=259,
    max_seq_len=0,
    num_memory_tokens=20,
    max_mem_len=100,
    attn_layers=Decoder(
        dim=512,
        depth=6,
        heads=4,
        rotary_pos_emb=True,
        shift_tokens=1,
        attn_flash=True,
        attn_onnxable=True,
        use_rmsnorm=True,
        sandwich_norm=True
    )
)

# Create input tensor (batch_size=1, sequence_length=1024, dim=512)
x = torch.randn(1, 1024, 512)

# Create mask tensor (batch_size=1, sequence_length=1024)
m = torch.randn(1, 1024) > 0

# Create initial mems (list of 6 tensors, each of shape [1, 100, 512])
mems = [torch.zeros(1, 100, 512) for _ in range(6)]

# Trigger the bug: pass mems and return_mems=True
logits, mems = net(x, mask=m, mems=mems, return_mems=True)

# Print output shapes to verify the bug
print("Logits shape:", logits.shape)
print("Updated mems shapes:", [m.shape for m in mems])