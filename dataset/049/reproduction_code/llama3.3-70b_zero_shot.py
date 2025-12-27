import torch
import torch.nn as nn

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self):
        super(RelPositionMultiHeadedAttention, self).__init__()
        self.u = torch.Tensor(1, 1)
        self.v = torch.Tensor(1, 1)

    def forward(self):
        return self.u.sum(), self.v.sum()

model = RelPositionMultiHeadedAttention()
print(model.forward())