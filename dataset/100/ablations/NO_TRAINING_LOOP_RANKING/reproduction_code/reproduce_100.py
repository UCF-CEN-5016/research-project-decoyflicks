import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

batch_size = 8
seq_length = 16
embedding_dim = 64

input_tensor = torch.randn(batch_size, seq_length, embedding_dim)
rotary_embedding = RotaryEmbedding(dim=embedding_dim, learned_freq=True, freqs_for='lang')

class SimpleModel(nn.Module):
    def __init__(self, rotary_embedding):
        super(SimpleModel, self).__init__()
        self.rotary_embedding = rotary_embedding

    def forward(self, x):
        return self.rotary_embedding.rotate_queries_with_cached_keys(x, x)

model = SimpleModel(rotary_embedding)
target_tensor = torch.randint(0, 2, (batch_size, seq_length))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for _ in range(2):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = loss_fn(output.view(-1, embedding_dim), target_tensor.view(-1))
    loss.backward()
    optimizer.step()

    if _ == 1:
        try:
            loss.backward()
        except RuntimeError as e:
            assert str(e) == "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward."