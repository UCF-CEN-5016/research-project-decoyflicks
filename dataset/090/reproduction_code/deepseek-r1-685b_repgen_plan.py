import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(10, 20)
        self.to_logits = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.embed(x)
        return self.to_logits(x)

# Setup model and optimizer
model = Encoder()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Sample data
X = torch.randn(32, 10)
y = torch.randn(32, 10)

# Training loop demonstrating the issue
for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(X)
    loss = nn.functional.mse_loss(outputs, y)
    loss.backward()
    
    # Check parameter updates
    before = model.to_logits.weight.detach().clone()
    optimizer.step()
    after = model.to_logits.weight.detach().clone()
    
    print(f"Epoch {epoch}: Weight changed = {not torch.allclose(before, after)}")
    print(f"Gradients exist: {model.to_logits.weight.grad is not None}")