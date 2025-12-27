import torch
import torch.nn.functional as F
import numpy as np

# Simulated rotary_emb module
class RotaryEmbedding:
    def rotate_queries_and_keys(self, Q, K, xpos):
        # Simplified rotation logic for demonstration
        # Actual implementation may vary
        Q_rotated = Q + xpos
        K_rotated = K + xpos
        return Q_rotated, K_rotated

# Minimal setup
if __name__ == "__main__":
    # Initialize tensors
    Q = torch.randn(1, 10, 512)  # Example shape
    K = torch.randn(1, 10, 512)  # Example shape
    xpos = torch.randn(1, 10, 512)  # Example shape, simulate xpos
    
    # Initialize rotary embedding
    rotary_emb = RotaryEmbedding()
    
    # Apply rotation
    Q_rotated, K_rotated = rotary_emb.rotate_queries_and_keys(Q, K, xpos)
    
    # Check for NaNs
    if torch.isnan(Q_rotated).any() or torch.isnan(K_rotated).any():
        print("NaNs detected in rotated queries or keys.")
    else:
        print("No NaNs detected.")
        
    # Example of checking NaNs in loss (assuming a simple loss function)
    loss = F.mse_loss(Q_rotated, K_rotated)
    if torch.isnan(loss):
        print("NaN detected in loss.")
    else:
        print("Loss is valid.")