import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionBug(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        attn = torch.einsum('b i d, b j d -> b i j', q, k) / (x.shape[-1] ** 0.5)
        # Bug: softmax along dim=1 instead of dim=2
        attn_bug = F.softmax(attn, dim=1)
        out_bug = torch.einsum('b i j, b j d -> b i d', attn_bug, v)
        out_bug = self.to_out(out_bug)
        return out_bug, attn_bug

class SelfAttentionCorrect(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        attn = torch.einsum('b i d, b j d -> b i j', q, k) / (x.shape[-1] ** 0.5)
        # Correct: softmax along dim=2 (over j)
        attn_correct = F.softmax(attn, dim=2)
        out_correct = torch.einsum('b i j, b j d -> b i d', attn_correct, v)
        out_correct = self.to_out(out_correct)
        return out_correct, attn_correct

torch.manual_seed(0)
batch, seq_len, dim = 2, 4, 8
x = torch.randn(batch, seq_len, dim)

attn_bug = SelfAttentionBug(dim)
attn_correct = SelfAttentionCorrect(dim)

out_bug, attn_map_bug = attn_bug(x)
out_correct, attn_map_correct = attn_correct(x)

print("Attention bug sums (dim=1 softmax):", attn_map_bug.sum(dim=1))
print("Attention correct sums (dim=2 softmax):", attn_map_correct.sum(dim=2))
print("Output difference norm:", (out_bug - out_correct).abs().sum().item())