import torch
import torch.nn as nn

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.u_bias = torch.Tensor(n_head, self.d_k)
        self.v_bias = torch.Tensor(n_head, self.d_k)

    def forward(self):
        return self.u_bias.sum(), self.v_bias.sum()

attn = RelPositionMultiHeadedAttention(n_head=8, d_model=64)
print(attn.forward())