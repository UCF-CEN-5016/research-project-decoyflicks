import torch

# Mock Gaussian diffusion class missing device property
class GaussianDiffusion:
    def __init__(self):
        self.betas = torch.randn(10)  # Just for demo
    
    # Missing device property that should be:
    # @property
    # def device(self):
    #     return self.betas.device

def offset_noise_strength(diffusion_model):
    # This would fail because it can't get device info
    return torch.randn(1, device=diffusion_model.device)

# Reproduction setup
def reproduce_bug():
    model = GaussianDiffusion()
    try:
        strength = offset_noise_strength(model)
        print("Success (unexpected!)")
    except AttributeError as e:
        print(f"Bug reproduced! Error: {e}")

reproduce_bug()