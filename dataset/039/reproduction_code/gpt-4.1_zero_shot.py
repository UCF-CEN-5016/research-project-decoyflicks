import torch

class RotaryPositionalEmbedding:
    def __init__(self, dim, max_seq_len=512):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)  # [1,1,max_seq_len,dim]
        self.sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)  # [1,1,max_seq_len,dim]

    def apply_rotary_pos_emb(self, x):
        x_rope = x.clone()
        neg_half_x = torch.cat((-x[..., 1::2], x[..., ::2]), dim=-1)
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

rpe = RotaryPositionalEmbedding(dim=4, max_seq_len=4)
x = torch.randn(3, 1, 4, 4)
rpe.apply_rotary_pos_emb(x)