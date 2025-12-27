import torch
import torch.nn as nn
import transformer_engine.pytorch as te

class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.te_layer = te.nn.Linear(16, 16, fp8=True)
        self.norm = te.nn.LayerNorm(16, eps=1e-5, fp8=True)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.te_layer(x)
        x = self.norm(x)
        return self.act(x)

model = SimpleTransformer().cuda().train()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
inp = torch.randn(4,16).cuda()
target = torch.randn(4,16).cuda()
for _ in range(10):
    opt.zero_grad()
    out = model(inp)
    loss = (out - target).pow(2).mean()
    loss.backward()
    opt.step()
    print(loss.item())