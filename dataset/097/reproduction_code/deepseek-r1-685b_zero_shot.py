import torch
import torch.nn as nn

class GaussianDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.betas = torch.randn(10)

def offset_noise_strength(diffusion):
    return torch.randn(1).to(diffusion.device)

diffusion = GaussianDiffusion()
offset_noise_strength(diffusion)