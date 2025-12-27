import torch
from torch import nn
import torch.optim as optim
from transformer_engine.pytorch import Linear
from transformer_engine.common.recipe import DelayedScaling, Format

class FP8Transformer(nn.Module):
    def __init__(self):
        super(FP8Transformer, self).__init__()
        self.layer = Linear(512, 512, params_dtype=torch.float8_e4m3fn)
        
    def forward(self, x):
        return self.layer(x)

fp8_recipe = DelayedScaling(margin=0, interval=1, fp8_format=Format.E4M3)

model = FP8Transformer().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(32, 512).cuda()
y = torch.randn(32, 512).cuda()

with torch.cuda.amp.autocast(enabled=True):
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x)
        loss = nn.functional.mse_loss(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item()}")
        if torch.isnan(loss):
            print("NaN detected!")
            break