import torch
from tqdm import tqdm

class GaussianDiffusion:
    def __init__(self):
        self.self_condition = False
    def model_predictions(self, x, t, self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        return None, None, None
    def ddim_sample(self, img, batch=1, device='cpu'):
        time_pairs = [(1, 0)]
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
            )
            
class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, clip_x_start=False):
        return None, None, None

LearnedGaussianDiffusion().ddim_sample(torch.tensor([0.0]))