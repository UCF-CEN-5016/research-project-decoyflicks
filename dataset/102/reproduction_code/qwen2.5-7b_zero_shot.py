import torch
import torch.nn as nn

class ResidualSimVQ(nn.Module):
    def __init__(self, channels, num_embeddings, embedding_dim, quantize_dropout=False):
        super(ResidualSimVQ, self).__init__()
        self.quantize_dropout = quantize_dropout
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.quantizer = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        if self.quantize_dropout and self.training:
            x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.quantizer(x)
        x = x.permute(0, 2, 1)
        return x

model = ResidualSimVQ(channels=1024, num_embeddings=512, embedding_dim=1024, quantize_dropout=True)
input_tensor = torch.randn(2, 1024, 17)
output = model(input_tensor)