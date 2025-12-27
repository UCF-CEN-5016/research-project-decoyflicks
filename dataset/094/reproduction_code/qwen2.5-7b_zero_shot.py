import torch

# Mock classes to simulate the model structure
class Decoder:
    def __init__(self, dim, depth, heads, rotary_pos_emb, shift_tokens, attn_flash, attn_onnxable, use_rmsnorm, sandwich_norm):
        pass  # Placeholder for actual implementation

class ContinuousTransformerWrapper(torch.nn.Module):
    def __init__(self, dim_out, max_seq_len, num_memory_tokens, max_mem_len, attn_layers):
        super(ContinuousTransformerWrapper, self).__init__()
        self.dim_out = dim_out
        self.max_seq_len = max_seq_len
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.attn_layers = attn_layers

    def forward(self, x, mask, mems, return_mems):
        # Simulate the bug: when return_mems is True, the code returns the original mems (not updated)
        # When return_mems is False, it returns the same mems (if provided) or None
        logits = x  # Placeholder for actual model output
        if return_mems:
            return logits, mems.clone()  # Fix: return a copy of mems
        else:
            return logits, mems if mems is not None else None

# Main simulation
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

x = torch.randn(1, 1024, 512)
mask = torch.randn(1, 1024) > 0  # Simulating a mask
mems = [torch.zeros(1, 100, 512) for _ in range(6)]  # Initial memory state

# Run the simulation with return_mems=True (the bug is triggered)
logits, updated_mems = net(x, mask=mask, mems=mems, return_mems=True)

# Output the results
print("Logits shape:", logits.shape)
print("Updated memory shapes:", [m.shape for m in updated_mems])

logits, updated_mems = net(x, mask=mask, mems=None, return_mems=True)

logits, updated_mems = net(x, mask=mask, mems=mems, return_mems=False)