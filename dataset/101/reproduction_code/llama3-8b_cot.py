import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RotaryEmbedding

# Minimal environment setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# Define the model and its components
class Transformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rotary_emb = RotaryEmbedding(dim)
    
    def forward(self, Q, K):
        return F.cosine_similarity(Q, K, dim=-1, eps=1e-6)

# Create the model and input tensors
model = Transformer(128).to(device)
Q = torch.randn(1, 100, 128).to(device)
K = torch.randn(1, 100, 128).to(device)

# Trigger the bug: rotate queries and keys
rotary_emb.rotate_queries_and_keys(Q, K)

# Verify the NaNs in K
print((K != K).any())