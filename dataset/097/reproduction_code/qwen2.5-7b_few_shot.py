import torch

class GaussianDiffusion:
    def __init__(self):
        self.betas = torch.tensor([1.0, 2.0, 3.0])  # Example betas
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def offset_noise_strength(self):
        return self.device

# Create an instance of GaussianDiffusion
diffusion = GaussianDiffusion()
print(diffusion.offset_noise_strength())