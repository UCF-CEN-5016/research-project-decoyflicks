import torch
import transformer_engine.pytorch as te

# Minimal transformer block with Transformer Engine FP8
class SimpleTransformer(te.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = te.nn.LayerNorm(10)
        self.linear1 = te.nn.Linear(10, 10, fp8=True)
        self.linear2 = te.nn.Linear(10, 10, fp8=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Setup
model = SimpleTransformer().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy data
X = torch.randn(8, 10, device='cuda')
y = torch.randn(8, 10, device='cuda')

# Mixed precision context for FP8
with te.fp8_autocast():
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = torch.nn.functional.mse_loss(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")