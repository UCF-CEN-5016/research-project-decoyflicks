import torch
import torch.nn as nn
import torch.optim as optim
from rotary_embedding_torch import RotaryEmbedding

batch_size = 8
seq_len = 64
feature_dim = 128

input_data = torch.randn(batch_size, seq_len, feature_dim)
target = torch.randint(0, 10, (batch_size, seq_len))

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.rotary_embedding = RotaryEmbedding(dim=feature_dim, learned_freq=True, freqs_for='lang')
        self.fc = nn.Linear(feature_dim, 10)

    def forward(self, x):
        x = self.rotary_embedding.forward(x)
        return self.fc(x)

model = SimpleModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# First training iteration
predictions = model(input_data)
loss = loss_fn(predictions.view(-1, 10), target.view(-1))
loss.backward()
optimizer.step()
optimizer.zero_grad()

# Second training iteration
predictions = model(input_data)
loss = loss_fn(predictions.view(-1, 10), target.view(-1))
loss.backward()  # This should raise RuntimeError