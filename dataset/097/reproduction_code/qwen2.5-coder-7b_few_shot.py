import torch
from typing import Optional

def _default_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GaussianDiffusion:
    def __init__(self, noise_schedule: Optional[torch.Tensor] = None, device: Optional[torch.device] = None):
        self.noise_schedule = noise_schedule if noise_schedule is not None else torch.tensor([1.0, 2.0, 3.0])
        self.device = device if device is not None else _default_device()

    def get_device(self) -> torch.device:
        return self.device

# Create an instance of GaussianDiffusion
diffusion = GaussianDiffusion()
print(diffusion.get_device())