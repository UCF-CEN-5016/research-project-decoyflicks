import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.to_logits = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = nn.Linear(4, 2)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(2, 4)
target = torch.randint(0, 2, (2,))

for _ in range(10):
    optimizer.zero_grad()
    out = model(x)
    loss = nn.functional.cross_entropy(out, target)
    loss.backward()
    optimizer.step()

print("encoder.to_logits.weight.grad:", model.encoder.to_logits.weight.grad)
print("encoder.to_logits.weight:", model.encoder.to_logits.weight)