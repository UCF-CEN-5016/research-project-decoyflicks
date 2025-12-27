import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_logits = nn.Linear(10, 10)

    def forward(self, x):
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.randn(5, 10)
y = torch.randn(5, 10)

for _ in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = (output - y).pow(2).sum()
    loss.backward()
    optimizer.step()

print(model.encoder.to_logits.weight.grad is not None)
print(model.encoder.to_logits.weight.data)