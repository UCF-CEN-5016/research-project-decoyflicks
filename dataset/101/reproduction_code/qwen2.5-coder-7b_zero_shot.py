import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.cos = nn.Parameter(torch.randn(self.dim))
        self.sin = nn.Parameter(torch.randn(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply elementwise cosine/sine to stored parameters and combine with input
        return torch.cat((x * torch.cos(self.cos), x * torch.sin(self.sin)), dim=-1)

def rotate_queries_and_keys(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    # mirror the original behavior: unsqueeze both tensors first
    queries, keys = queries.unsqueeze(0), keys.unsqueeze(0)
    neg_pos = keys.new_tensor([[-1.], [1.]])
    pos_neg = keys.new_tensor([[1.], [-1.]])
    sin_rot = keys[:, :, 3:] * neg_pos
    cos_rot = keys[:, :, :3] * pos_neg
    return torch.cat((sin_rot, cos_rot), dim=2).squeeze()

if __name__ == "__main__":
    Q = torch.randn(4, 8)
    K = torch.randn(4, 8)

    rotary_emb = RotaryEmbedding(4)
    Q = rotary_emb(Q)
    K = K.unsqueeze(0)
    K = rotate_queries_and_keys(Q, K)
    print(K.isnan().any())