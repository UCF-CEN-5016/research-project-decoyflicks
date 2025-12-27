import torch

class GaussianDiffusion:
    @property
    def device(self):
        return self.betas.device  # This method is missing in our case

# Example usage (would fail due to missing device property)
diffusion = GaussianDiffusion()
offset_noise_strength(diffusion)  # This would raise an error