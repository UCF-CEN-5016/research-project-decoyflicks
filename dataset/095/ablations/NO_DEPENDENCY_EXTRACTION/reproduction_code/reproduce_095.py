import torch
from tqdm import tqdm
from denoising_diffusion_pytorch import GaussianDiffusion, LearnedGaussianDiffusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
height, width = 64, 64
img = torch.randn(batch_size, 3, height, width, device=device)

timesteps = 1000
sampling_timesteps = 100
model = LearnedGaussianDiffusion(image_size=(height, width), timesteps=timesteps)

try:
    model.ddim_sample(shape=(batch_size, 3, height, width), return_all_timesteps=False)
except TypeError as e:
    print(e)