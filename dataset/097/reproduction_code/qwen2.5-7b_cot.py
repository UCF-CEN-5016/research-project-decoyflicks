import torch

class GaussianDiffusion:
    def __init__(self):
        # Simulate betas tensor on a specific device
        self.betas = torch.tensor([0.001, 0.01, 0.1], device='cuda')  # Assume CUDA device

    @property
    def device(self):
        return self.betas.device

    def offset_noise_strength(self):
        device = self.device
        print(f"Using device: {device}")

# Trigger the bug by calling the method
diffusion = GaussianDiffusion()
diffusion.offset_noise_strength()