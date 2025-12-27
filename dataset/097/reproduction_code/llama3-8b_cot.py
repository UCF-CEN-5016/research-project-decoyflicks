import torch

class GaussianDiffusion(torch.nn.Module):
    def __init__(self):
        super(GaussianDiffusion, self).__init__()
        self.betas = torch.randn(1)

    @property
    def device(self):
        return self.betas.device

# Triggering conditions:
diffusion = GaussianDiffusion()
print(diffusion.offset_noise_strength)  # Expected to raise an error