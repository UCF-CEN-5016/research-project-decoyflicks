self.model_predictions(img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True)

def model_predictions(self, x, t, clip_x_start, rederive_pred_noise):

import torch
from torch import nn

class GaussianDiffusion:
    def ddim_sample(self):
        img = torch.randn(1, 3, 64, 64)
        time_cond = torch.tensor([100], device='cuda')
        self_cond = torch.randn(1, 3, 64, 64)
        
        # This line will raise the error
        self.model_predictions(img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True)

class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, clip_x_start, rederive_pred_noise):
        # This function has the same parameter name as the keyword argument
        # This leads to a conflict
        pass

# This will raise the error
diffusion = LearnedGaussianDiffusion()
diffusion.ddim_sample()

class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, clip_x_start_flag, rederive_pred_noise):
        pass

import torch
from torch import nn

class GaussianDiffusion:
    def ddim_sample(self):
        img = torch.randn(1, 3, 64, 64)
        time_cond = torch.tensor([100], device='cuda')
        self_cond = torch.randn(1, 3, 64, 64)
        
        self.model_predictions(img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True)

class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, clip_x_start_flag, rederive_pred_noise):
        pass

diffusion = LearnedGaussianDiffusion()
diffusion.ddim_sample()