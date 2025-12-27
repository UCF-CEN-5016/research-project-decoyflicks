import torch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer
from random import random

torch.manual_seed(42)

batch_size = 4
height, width = 64, 64
x = torch.rand(batch_size, 3, height, width)
t = torch.randint(0, 1000, (batch_size,))

model = Unet()
diffusion_model = GaussianDiffusion(model)
diffusion_model.self_condition = True

if diffusion_model.self_condition and random() < 0.5:
    with torch.inference_mode():
        x_self_cond = diffusion_model.model_predictions(x.clone().detach(), t).pred_x_start
        x_self_cond = x_self_cond.detach_()