import torch
import torch.nn as nn

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        # Incorrect: uninitialized bias terms using torch.Tensor
        self.u_bias = torch.Tensor(n_head, self.d_head)  # uninitialized tensor
        self.v_bias = torch.Tensor(n_head, self.d_head)  # uninitialized tensor

    def forward(self, x):
        # Just return sum to show effect of uninitialized parameters
        return self.u_bias.sum() + self.v_bias.sum()

# Instantiate and run
mha = RelPositionMultiHeadedAttention(n_head=4, d_model=16)

print("u_bias sum:", mha.u_bias.sum().item())  # May print nan or random values
print("v_bias sum:", mha.v_bias.sum().item())