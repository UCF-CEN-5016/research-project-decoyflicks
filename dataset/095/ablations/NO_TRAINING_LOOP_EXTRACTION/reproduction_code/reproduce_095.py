import torch
from denoising_diffusion_pytorch import GaussianDiffusion, LearnedGaussianDiffusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
img = torch.randn(batch_size, 3, 64, 64).to(device)
time_pairs = torch.randn(10, 2).to(device)

class CustomLearnedGaussianDiffusion(LearnedGaussianDiffusion):
    def model_predictions(self, img, time_cond, self_cond=None, clip_x_start=True, rederive_pred_noise=False):
        return super().model_predictions(img, time_cond, self_cond, clip_x_start, rederive_pred_noise)

ldm = CustomLearnedGaussianDiffusion()

try:
    ldm.ddim_sample(img, time_pairs)
except TypeError as e:
    print(e)