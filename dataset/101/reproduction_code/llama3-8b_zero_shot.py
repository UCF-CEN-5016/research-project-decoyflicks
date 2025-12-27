import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.cos = nn.Parameter(torch.randn(self.dim))
        self.sin = nn.Parameter(torch.randn(self.dim))

    def forward(self, x):
        return torch.cat((x * F.cos(self.cos), x * F.sin(self.sin)), dim=-1)

def rotate_queries_and_keys(Q, K):
    Q, K = Q.unsqueeze(0), K.unsqueeze(0)
    sin_rot = K[:, :, 3:] * torch.tensor([[-1.], [1.]])
    cos_rot = K[:, :, :3] * torch.tensor([[1.], [-1.]])
    return torch.cat((sin_rot, cos_rot), dim=2).squeeze()

Q = torch.randn(4, 8)
K = torch.randn(4, 8)

rotary_emb = RotaryEmbedding(4)
Q = rotary_emb(Q)
K = K.unsqueeze(0)
K = rotate_queries_and_keys(Q, K)
print(K.isnan().any())