import torch

class GaussianDiffusion:
    def __init__(self, betas):
        self.betas = betas

    def offset_noise_strength(self, t, strength):
        return self.betas[t] * strength

class ClassifierFreeGuidance:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    def forward(self, t, strength):
        return self.diffusion.offset_noise_strength(t, strength)

# setup
betas = torch.randn(10)
diffusion = GaussianDiffusion(betas)
guidance = ClassifierFreeGuidance(diffusion)

# reproduce issue
try:
    guidance.forward(0, 1.0)
except AttributeError as e:
    print(e)