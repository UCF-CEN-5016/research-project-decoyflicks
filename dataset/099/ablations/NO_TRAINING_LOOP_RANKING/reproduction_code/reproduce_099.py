import torch
import torch.nn as nn
from rotary_embedding_torch.rotary_embedding_torch import RotaryEmbedding

device = 'cuda'
batch_size = 8
seq_length = 128
embedding_dim = 64

queries = torch.randn(batch_size, seq_length, embedding_dim, device=device)
keys = torch.randn(batch_size, seq_length, embedding_dim, device=device)

rotary_embedding = RotaryEmbedding(dim=embedding_dim, learned_freq=True, use_xpos=True, cache_if_possible=True)

class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.rotary_embedding = rotary_embedding
        self.fc = nn.Linear(embedding_dim, 10)  # Assuming 10 classes for classification

    def forward(self, x):
        x = self.rotary_embedding.rotate_queries_and_keys(x, x)
        return self.fc(x)

model = SimpleTransformer().to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

targets = torch.randint(0, 10, (batch_size,), device=device)

outputs = model(queries)
loss = criterion(outputs.view(-1, 10), targets)

if torch.isnan(loss):
    print("NaN loss detected")

loss.backward()

for param in model.parameters():
    if torch.isnan(param.grad).any():
        print("NaN gradient detected")