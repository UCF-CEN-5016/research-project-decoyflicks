import torch
from torch import nn
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, self_condition=False):
        super().__init__()
        self.self_condition = self_condition

    def model_predictions(self, img, time_cond, self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        return None, None, None, None

    def ddim_sample(self, shape, device):
        batch = shape[0]
        img = torch.randn(shape, device=device)
        time_pairs = [(1, 2)]
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = None if not self.self_condition else torch.zeros_like(img)
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
            )
        return img

class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, clip_x_start=False):
        return None, None, None, None

diffusion = LearnedGaussianDiffusion()
diffusion.ddim_sample((1, 3, 32, 32), device='cpu')