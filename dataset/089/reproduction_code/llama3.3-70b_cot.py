import torch
import torch.nn as nn

class CustomTransformer(nn.Module):
    def __init__(self, dim_head, heads, kv_heads, qk_norm):
        super(CustomTransformer, self).__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.kv_heads = kv_heads
        self.qk_norm = qk_norm
        
        # Original line of code
        # self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
        
        # Suggested fix
        self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

    def forward(self, x):
        # Simplified forward pass for demonstration
        if self.qk_norm and self.kv_heads != self.heads:
            # Here we would use self.qk_norm_k_scale in a way that triggers the bug
            # For demonstration, let's just print the shape
            print(f"qk_norm_k_scale shape: {self.qk_norm_k_scale.shape}")
            # The bug would be triggered by operations involving self.qk_norm_k_scale
            # and other tensors with incompatible shapes due to kv_heads != heads
        return x

# Minimal setup to trigger the bug
if __name__ == "__main__":
    dim_head = 64
    heads = 8
    kv_heads = 16  # Different from heads to trigger the bug
    qk_norm = True
    
    model = CustomTransformer(dim_head, heads, kv_heads, qk_norm)
    input_tensor = torch.randn(1, 10, dim_head * heads)  # Example input
    output = model(input_tensor)