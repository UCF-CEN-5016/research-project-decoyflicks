import torch
import torch.nn as nn

# Define a model that accepts 'text' and 'image_codes' arguments only
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
    
    def forward(self, text, image_codes):
        return self.linear(torch.cat([text, image_codes], dim=-1))

# Create dummy data
text = torch.randn(32, 10)
image_codes = torch.randn(32, 5)

# Training loop that uses the model
model = SimpleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    loss = model(text, image_codes)  # Pass only 'text' and 'image_codes' to the model
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")