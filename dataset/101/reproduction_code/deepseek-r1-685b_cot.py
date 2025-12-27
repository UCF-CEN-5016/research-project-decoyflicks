import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.xpos_scale = None

    def rotate_queries_and_keys(self, q, k):
        batch, seq_len, dim = q.shape
        
        # Generate position indices
        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        
        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Apply xpos scaling if enabled
        if self.xpos_scale is not None:
            scale = self.xpos_scale ** torch.arange(seq_len, device=q.device)
            scale = scale[:, None]
            emb = emb * scale
        
        # Expand dimensions for broadcasting
        cos = torch.cos(emb)[:, None, :]
        sin = torch.sin(emb)[:, None, :]
        
        # Rotate queries and keys
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot
    
    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

# Test setup
dim = 64
seq_len = 1024
batch_size = 4

# Create module with xpos enabled
rotary_emb = RotaryEmbedding(dim)
rotary_emb.xpos_scale = 1.02  # Enable xpos scaling

# Create random Q, K tensors
Q = torch.randn(batch_size, seq_len, dim)
K = torch.randn(batch_size, seq_len, dim)

# Apply rotary embeddings
Q_rot, K_rot = rotary_emb.rotate_queries_and_keys(Q, K)

# Check for NaNs
print("NaNs in Q_rot:", torch.isnan(Q_rot).any())
print("NaNs in K_rot:", torch.isnan(K_rot).any())
print("Number of NaNs in K_rot:", torch.isnan(K_rot).sum().item())