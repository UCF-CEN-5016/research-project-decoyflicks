import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_embedding = nn.Embedding(10, 5)

    def forward(self, num_patches):
        tokens = torch.zeros(1, 5)
        return tokens + self.pos_embedding.weight[:, 1:(num_patches + 1)]

encoder = Encoder()
print(encoder(5))