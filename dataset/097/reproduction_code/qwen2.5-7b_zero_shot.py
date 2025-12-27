import torch

class GaussianDiffusion:
    def __init__(self):
        self.device = 'cuda'
        self.betas = torch.tensor([1.0, 2.0], device=self.device)

    def offset_noise_strength(self):
        return self.device

diffusion = GaussianDiffusion()
diffusion.offset_noise_strength()