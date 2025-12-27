import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(10, 20)
        self.to_logits = nn.Linear(20, 10)  # Problematic layer
        
    def forward(self, x):
        # Forgot to use to_logits in forward pass
        return self.embed(x)  # Should be: return self.to_logits(self.embed(x))

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
    before = model.to_logits.weight.clone()
    optimizer.step()
    after = model.to_logits.weight.clone()
    
    print(f"Epoch {epoch}: Weight changed = {not torch.allclose(before, after)}")
    print(f"Gradients exist: {model.to_logits.weight.grad is not None}")