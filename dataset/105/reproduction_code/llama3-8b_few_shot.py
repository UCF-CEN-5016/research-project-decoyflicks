import torch
import random

# Reproduce the error-inducing setup
class MyResidualVQ(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vq = ResidualVQ(**kwargs)

    def forward(self, x):
        z = x  # sample input data
        return self.vq(z)

model = MyResidualVQ(
    dim=512,
    num_quantizers=2,
    codebook_size=16 * 1024,
    stochastic_sample_codes=True,
    shared_codebook=True,
    commitment_weight=1.0,
    kmeans_init=True,
    threshold_ema_dead_code=2,
    quantize_dropout=True,
    quantize_dropout_cutoff_index=1,
    quantize_dropout_multiple_of=1,
)

# Reproduce the error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
for _ in range(10):
    x = torch.randn(32, 512).to(device)
    z_tilde, _, commit_loss = model(x)
print(f"Commit loss: {commit_loss.item()}")