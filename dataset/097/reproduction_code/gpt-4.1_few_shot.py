import torch

class GaussianDiffusion:
    def __init__(self):
        self.betas = torch.linspace(0.1, 0.2, steps=10)

    # Missing device property causes error when accessing device-dependent methods
    # @property
    # def device(self):
    #     return self.betas.device

    def offset_noise_strength(self):
        # This method assumes self.device exists to place tensors on correct device
        noise = torch.randn(3, device=self.device)  # Will raise AttributeError
        return noise * self.betas[0]

# Instantiate and move to CUDA
gd = GaussianDiffusion()
gd.betas = gd.betas.cuda()

try:
    noise_strength = gd.offset_noise_strength()
    print("Noise strength:", noise_strength)
except AttributeError as e:
    print("Error:", e)