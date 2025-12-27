import torch
from torch import nn, Tensor

class GaussianDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
    def model_predictions(self, x: Tensor, t: int, clip_x_start: bool = False) -> (Tensor, Tensor):
        return x, x, 1, True, True

    def ddim_sample(self, img: Tensor, batch: int)
        time_pairs = [(i, i+1) for i in range(batch)]
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device='cuda', dtype=torch.long)
            self_cond = img if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True)

class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x: Tensor, t: int, clip_x_start: bool = False) -> (Tensor,):
        return x,

import torch.nn.functional as F
from tqdm import tqdm

device = 'cuda'
model = LearnedGaussianDiffusion()
img = torch.randn(1, 3, 256, 256).to(device)
batch = 10
model.ddim_sample(img, batch)