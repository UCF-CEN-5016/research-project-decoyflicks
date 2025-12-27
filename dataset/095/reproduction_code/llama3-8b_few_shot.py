import torch
from your_module import LearnedGaussianDiffusion

# Minimal setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LearnedGaussianDiffusion()
batch = 1
img = torch.randn(batch, 64, 64, device=device)

# Run into the error
time_pairs = [(0, 100)]
for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
    time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
    self_cond = img if not hasattr(model, 'self_condition') else None
    pred_noise, x_start, *_ = model.model_predictions(
        img, time_cond, self_cond, clip_x_start=True, rederive_pred_noise=True
    )