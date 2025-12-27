import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
sequence_length = 128
embedding_dim = 64

q = torch.randn(batch_size, sequence_length, embedding_dim, device=device)
k = torch.randn(batch_size, sequence_length, embedding_dim, device=device)

model = RotaryEmbedding(dim=embedding_dim, use_xpos=True).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for _ in range(10):
    output_q, output_k = model.rotate_queries_with_cached_keys(q, k)
    target = torch.randn_like(output_q)
    loss = nn.MSELoss()(output_q, target)
    
    loss.backward()
    optimizer.step()
    
    print(f'Loss: {loss.item()}')
    assert torch.isnan(loss.item()), "Loss is not NaN"