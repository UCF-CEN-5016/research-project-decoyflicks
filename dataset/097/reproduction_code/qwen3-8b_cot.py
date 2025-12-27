import torch

class GaussianDiffusion:
    def __init__(self):
        # Simulate betas tensor on a specific device
        self.betas = torch.tensor([0.001, 0.01, 0.1], device='cuda')  # Assume CUDA device

    def offset_noise_strength(self):
        # This method relies on the `device` property to access the device
        # However, the property is missing in the buggy implementation
        device = self.device  # This will raise an AttributeError
        print(f"Using device: {device}")

# Trigger the bug by calling the method
diffusion = GaussianDiffusion()
diffusion.offset_noise_strength()