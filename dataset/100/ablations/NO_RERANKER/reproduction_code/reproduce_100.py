import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

torch.manual_seed(42)

batch_size = 8
seq_len = 10
embedding_dim = 64
input_tensor = torch.rand(batch_size, seq_len, embedding_dim)

rotary_embedding = RotaryEmbedding(dim=embedding_dim, learned_freq=False, freqs_for='lang')
loss_fn = nn.MSELoss()
target_tensor = torch.rand_like(input_tensor)

output = rotary_embedding(input_tensor)
loss = loss_fn(output, target_tensor)

loss.backward()

optimizer = torch.optim.SGD(rotary_embedding.parameters(), lr=0.01)
optimizer.step()

output_second = rotary_embedding(input_tensor)
loss_second = loss_fn(output_second, target_tensor)

try:
    loss_second.backward()
except RuntimeError as e:
    assert str(e) == "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."