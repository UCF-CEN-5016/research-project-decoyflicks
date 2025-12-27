import torch
import torch.nn as nn

# Define a model that accepts 'text' and 'image_codes' arguments only
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # match the concatenated size of text (10) and image_codes (5) => 15
        self.linear = nn.Linear(15, 2)
    
    def forward(self, text, image_codes):
        return self.linear(torch.cat([text, image_codes], dim=-1))

# Create dummy data
text = torch.randn(32, 10)
image_codes = torch.randn(32, 5)
mask = torch.randint(0, 2, (32, 10))  # a mask that will be passed as a keyword arg

# Training loop that uses the model
model = SimpleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(10):
    optimizer.zero_grad()
    # Intentionally pass an unexpected keyword argument 'mask' to reproduce the TypeError
    output = model(text, image_codes, mask=mask)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")