import torch
from classifier_free_guidance import GaussianDiffusion

torch.manual_seed(42)
batch_size = 4
betas = torch.rand(batch_size, 1)

gaussian_diffusion = GaussianDiffusion(betas)

try:
    gaussian_diffusion.offset_noise_strength()
except AttributeError as e:
    print(f"Exception: {e}")

from denoising_diffusion_pytorch.version import __version__
print(f"Library version: {__version__}")