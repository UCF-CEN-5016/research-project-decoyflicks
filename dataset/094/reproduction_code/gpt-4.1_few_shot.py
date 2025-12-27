import torch
from some_module import ContinuousTransformerWrapper, Decoder  # Replace 'some_module' with actual module name

# Initialize model with mems-related settings
net = ContinuousTransformerWrapper(
    dim_out             = 259,
    max_seq_len         = 0,
    num_memory_tokens   = 20,
    max_mem_len         = 100,
    attn_layers = Decoder (
        dim             = 512,
        depth           = 6,
        heads           = 4,
        rotary_pos_emb  = True,
        shift_tokens    = 1,
        attn_flash      = True,
        attn_onnxable   = True,
        use_rmsnorm     = True,
        sandwich_norm   = True
    )
)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]

# Calling with mems passed in and return_mems=True triggers bug
logits, returned_mems = net(x, mask=m, mems=mems, return_mems=True)

print("Logits shape:", logits.shape)
print("Returned mems shapes:", [mem.shape for mem in returned_mems])