import torch
from transformers import ContinuousTransformerWrapper, Decoder

# Define a minimal setup for the bug reproduction
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

# Set up input data
x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]

# Test the model with return_mems=True (buggy behavior)
logits, mems_expected = net(x, mask=m, mems=mems, return_mems=True)
print(logits.shape)  # Should print the shape of the logits
print([m.shape for m in mems_expected])  # Should print the shapes of the memorized tokens

# Test the model with return_mems=None (working behavior)
logits, _ = net(x, mask=m, mems=mems, return_mems=None)
print(logits.shape)  # Should print the shape of the logits