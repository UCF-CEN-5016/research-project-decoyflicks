import torch
import torch.nn as nn

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__()
        self.P = 2 ** 12
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, d_model)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, d_model)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        key_pos_emb = self.key_pos_embeddings[self.P - key.shape[0]:self.P + query.shape[0]]
        query_pos_bias = self.query_pos_bias[None, None, :, :]
        ac = torch.einsum('ibhd,jbhd->ijbh', query + query_pos_bias, key)
        return ac

batch_size = 4
feature_dim = 4
x = torch.rand(batch_size, 4, 3, feature_dim)

attention = RelativeMultiHeadAttention(heads=2, d_model=feature_dim, dropout_prob=0.1)
attention.key_pos_embeddings.data = torch.zeros((2**12 * 2, 2, feature_dim))
attention.key_pos_bias.data = torch.zeros((2**12 * 2, 2))
attention.query_pos_bias.data = torch.zeros((2, feature_dim))

x_rope = attention.get_scores(x, x)

cos_cached = torch.rand(4, 2, 2, feature_dim)
sin_cached = torch.rand(4, 2, 2, feature_dim)
neg_half_x = -x

try:
    x_rope = (x_rope * cos_cached[:x.shape[0]]) + (neg_half_x * sin_cached[:x.shape[0]])
except RuntimeError as e:
    print(e)
    print(f"x_rope shape: {x_rope.shape}, cos_cached shape: {cos_cached[:x.shape[0]].shape}, sin_cached shape: {sin_cached[:x.shape[0]].shape}")