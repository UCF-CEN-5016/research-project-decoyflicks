import torch

class GaussianDiffusion:
    def __init__(self):
        self.betas = torch.tensor([0.1, 0.2])

    def offset_noise_strength(self):
        return self.betas.device

diff = GaussianDiffusion()
print(diff.offset_noise_strength())