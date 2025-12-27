import torch

class GaussianDiffusion:
    def __init__(self, betas, device='cpu'):
        self.betas = betas
        self.device = device

def offset_noise_strength(diffusion_model):
    return torch.randn(1, device=diffusion_model.device)

# Create model with CUDA tensors
betas = torch.randn(10, device='cuda')
model = GaussianDiffusion(betas, device='cuda')

try:
    strength = offset_noise_strength(model)
    print(f"Noise strength: {strength.item()}")
except AttributeError as e:
    print(f"Error: {e} (missing device property)")