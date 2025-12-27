import torch

class GaussianDiffusion:
    @property
    def device(self):
        return self.betas.device
    
    def offset_noise_strength(self, t: int) -> float:
        # Example implementation using the beta tensor's device
        return self.betas[t].item()

diffusor = GaussianDiffusion()