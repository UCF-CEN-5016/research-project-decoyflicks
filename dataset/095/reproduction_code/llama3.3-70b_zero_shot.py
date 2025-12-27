import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = nn.Linear(10, 10)
        self.self_condition = False

    def model_predictions(self, img, time_cond, self_cond, clip_x_start, rederive_pred_noise):
        return torch.randn(1), torch.randn(1), torch.randn(1)

    def ddim_sample(self, batch, x_start):
        device = self.device
        time_pairs = [(1, 2), (2, 3)]
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                torch.randn(1), time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
            )

class LearnedGaussianDiffusion(GaussianDiffusion):
    def model_predictions(self, x, t, clip_x_start=False):
        return torch.randn(1), torch.randn(1)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion = LearnedGaussianDiffusion(device)
    try:
        diffusion.ddim_sample(1, torch.randn(1))
    except TypeError as e:
        print(e)

if __name__ == "__main__":
    main()