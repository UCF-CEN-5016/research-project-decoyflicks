import torch
from torch.nn import Linear
import torch.nn.functional as F

# Minimal code to reproduce and fix the NaN bug with Xpos in transformers
def rotate_queries_and_keys(Q, K):
    """Applies rotation for relative positioning."""
    device = Q.device
    seq_len = Q.size(1)
    
    # Ensure correct handling of edge cases
    Q = Q * (K != 0).float()
    K = K * (K != 0).float() + (K == 0).float()

    angle = torch.arange(seq_len, device=device) * 10.0 / seq_len
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    Q = F.linear(Q, cos.view(1, -1, 1))
    K = F.linear(K, sin.view(1, -1, 1))

    return Q, K

def main():
    # Create example tensors
    batch_size = 2
    seq_len = 5
    d_k = 64
    
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    
    # Apply the rotation with proper handling of zeros
    Q_rot, K_rot = rotate_queries_and_keys(Q, K)
    
    print("After rotation, shape of K:", K_rot.shape)
    print("Any NaNs in K?", K_rot.isnan().any())

if __name__ == "__main__":
    main()