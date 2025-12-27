import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = te.Linear(128, 128, params_dtype=torch.float8_e4m3fn)
    
    def forward(self, x):
        return self.linear(x)

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()
fp8_recipe = recipe.DelayedScaling()

x = torch.randn(32, 128).cuda()
y = torch.randn(32, 128).cuda()

with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

print(loss.item())