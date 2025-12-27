import torch
import torch.nn as nn

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        # Initialize bias terms properly
        self.u_bias = nn.Parameter(torch.randn(n_head, self.d_k))  
        self.v_bias = nn.Parameter(torch.randn(n_head, self.d_k))  
    
    def forward(self, x):
        # Simplified forward pass to demonstrate the issue
        return self.u_bias.sum() + self.v_bias.sum()

# Create attention module
attention = RelPositionMultiHeadedAttention(d_model=64, n_head=8)

# Run multiple times to show non-deterministic behavior
for i in range(5):
    output = attention(torch.randn(1, 10, 64))
    print(f"Run {i}: Output = {output.item()}")