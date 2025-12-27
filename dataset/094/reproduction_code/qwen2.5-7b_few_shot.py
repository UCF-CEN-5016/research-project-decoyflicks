import torch

# Custom transformer wrapper with improved structure
class ContinuousTransformerWrapper(torch.nn.Module):
    def __init__(self, dim_out, max_seq_len, num_memory_tokens, max_mem_len, attn_layers):
        super().__init__()
        self.dim_out = dim_out
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.attn_layers = attn_layers

    def forward(self, x, mask, mems, return_mems):
        # Simulated forward pass (simplified version)
        if mems is not None:
            mems = self.update_mems(x, mems)
        return x, mems

    def update_mems(self, x, mems):
        # Placeholder for updating mems based on input x
        return mems  # Placeholder logic

# Reproduction code with improved structure
net = ContinuousTransformerWrapper(
    dim_out=259,
    max_seq_len=0,
    num_memory_tokens=20,
    max_mem_len=100,
    attn_layers=torch.nn.ModuleDict({
        'Decoder': torch.nn.ModuleDict({
            'dim': 512,
            'depth': 6,
            'heads': 4,
            'rotary_pos_emb': True,
            'shift_tokens': 1,
            'attn_flash': True,
            'attn_onnxable': True,
            'use_rmsnorm': True,
            'sandwich_norm': True
        })
    })
)

x = torch.randn(1, 1024, 512)
m = torch.randn(1, 1024) > 0
mems = [torch.zeros(1, 100, 512) for _ in range(6)]
logits, mems = net(x, mask=m, mems=mems, return_mems=True)
print("Logits shape:", logits.shape)
print("Mems shapes:", [m.shape for m in mems])