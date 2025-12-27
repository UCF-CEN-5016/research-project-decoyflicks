import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSimVQ(nn.Module):
    def __init__(self, dim, num_quantizers, quantizer_dim, quantize_dropout):
        super().__init__()
        self.quantizers = nn.ModuleList([nn.Linear(dim, quantizer_dim) for _ in range(num_quantizers)])
        self.quantize_dropout = quantize_dropout

    def forward(self, x):
        should_quantize_dropout = self.training and self.quantize_dropout and not return_loss
        all_losses = []
        all_indicies = []
        for quantizer in self.quantizers:
            z = quantizer(x)
            all_losses.append(F.mse_loss(z, x))
            all_indicies.append(torch.randint(0, 10, (x.shape[0], x.shape[1])))
        return all_losses, all_indicies

x = torch.randn(2, 17, 1024)
model = ResidualSimVQ(1024, 9, 1024, True)
print(model(x))