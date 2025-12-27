import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Model setup with FP8
class FP8Transformer(torch.nn.Module):
    def __init__(self):
        super(FP8Transformer, self).__init__()
        self.layer = te.Linear(512, 512, params_dtype=torch.float8_e4m3fn)
        
    def forward(self, x):
        return self.layer(x)

# FP8 training recipe
fp8_recipe = recipe.DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=recipe.Format.E4M3
)

# Model and optimizer
model = FP8Transformer().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Sample data
x = torch.randn(32, 512).cuda()
y = torch.randn(32, 512).cuda()

# Training loop that produces NaN loss
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x)
        loss = torch.nn.functional.mse_loss(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        if torch.isnan(loss).any():
            print("NaN detected!")
            break