import torch

# Minimal GaussianDiffusion class mimicking classifier_free_guidance.py without device property
class GaussianDiffusion:
    def __init__(self, betas):
        self.betas = betas

    # Missing device property here causes failure when calling offset_noise_strength

    def offset_noise_strength(self):
        # This method uses self.device, which is missing
        # Simulate usage that triggers the bug:
        noise = torch.randn_like(self.betas).to(self.device)  # This line will fail
        return noise * 0.5

# Setup: create betas tensor on CUDA (if available), to trigger device mismatch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
betas = torch.linspace(0.0001, 0.02, steps=10).to(device)

gd = GaussianDiffusion(betas)

try:
    # This will raise AttributeError because .device property is missing
    noise_strength = gd.offset_noise_strength()
except AttributeError as e:
    print(f"Caught expected AttributeError: {e}")

# Fix by adding the missing device property:
class GaussianDiffusionFixed:
    def __init__(self, betas):
        self.betas = betas

    @property
    def device(self):
        return self.betas.device

    def offset_noise_strength(self):
        noise = torch.randn_like(self.betas).to(self.device)
        return noise * 0.5

gd_fixed = GaussianDiffusionFixed(betas)
noise_strength_fixed = gd_fixed.offset_noise_strength()
print("Fixed code output:", noise_strength_fixed)