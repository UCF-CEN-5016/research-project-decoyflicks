import torch
import torch.nn as nn
from rotary_embedding_torch.rotary_embedding_torch import (
    apply_rotary_emb,
    RotaryEmbedding,
    apply_learned_rotations,
    broadcat
)

device = 'cuda'
batch_size = 32
sequence_length = 64
embedding_dimension = 512

input_data = torch.randn(batch_size, sequence_length, embedding_dimension).to(device)
target_labels = torch.randint(0, 100, (batch_size, sequence_length)).to(device)

model = nn.Transformer(d_model=embedding_dimension, nhead=8).to(device)
model.train()

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    optimizer.zero_grad()
    output = model(input_data)
    loss = loss_function(output.view(-1, output.size(-1)), target_labels.view(-1))
    
    if torch.isnan(loss):
        print('Loss is NaN, bug reproduced.')
    
    loss.backward()
    optimizer.step()