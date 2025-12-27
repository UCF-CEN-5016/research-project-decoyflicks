import torch

# Custom transformer wrapper (simplified for demonstration)
class ContinuousTransformerWrapper(torch.nn.Module):
    def __init__(self, dim_out, max_seq_len, num_memory_tokens, max_mem_len, attn_layers):
        super().__init__()
        self.dim_out = dim_out
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.attn_layers = attn_layers

    def forward(self, x, mask, mems, return_mems):
        # Simulated forward pass (this is a simplified version)
        # In a real scenario, the model would process x and update mems
        # However, the bug occurs when return_mems is True and mems is provided
        if mems is not None:
            # If mems is provided, the function should return updated mems
            # But the bug causes it to not return properly
            pass  # Simulated logic that fails to return updated mems
        return x, mems  # This is a simplified return, but the actual implementation may have the bug

# Reproduction code
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

import torch

# Simulated ContinuousTransformerWrapper (simplified for demonstration)
class ContinuousTransformerWrapper(torch.nn.Module):
    def __init__(self, dim_out, max_seq_len, num_memory_tokens, max_mem_len, attn_layers):
        super().__init__()
        self.dim_out = dim_out
        self.num_memory_tokens = num_memory_tokens
        self.max_mem_len = max_mem_len
        self.attn_layers = attn_layers

    def forward(self, x, mask, mems, return_mems):
        # Simulated forward pass (this is a simplified version)
        # In a real scenario, the model would process x and update mems
        # However, the bug occurs when return_mems is True and mems is provided
        # The function fails to return updated mems when mems is not None
        # This is a simplified return to demonstrate the issue
        if mems is not None:
            # This is a placeholder for the actual logic that may fail to update mems
            pass  # Simulated logic that fails to return updated mems
        return x, mems  # This is a simplified return, but the actual implementation may have the bug

# Reproduction code
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