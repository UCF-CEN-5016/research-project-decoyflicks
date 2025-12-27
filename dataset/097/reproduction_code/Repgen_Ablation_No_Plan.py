import torch
from tqdm import tqdm
import torch.nn.functional as F

def normalize_to_neg_one_to_one(images):
    return images / (images.abs().max() + 1e-7)

def unnormalize_to_zero_to_one(images):
    return (images + 1) * 0.5

def default(val, default_val):
    return val if val is not None else default_val

class GaussianDiffusion:
    def __init__(self, sigma_min=1e-4, sigma_max=20., device='cpu'):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device
    
    def sample_schedule(self, num_sample_steps=50):
        sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_sample_steps).to(self.device)
        return sigmas
    
    @torch.no_grad()
    def sample(self, batch_size=16, num_sample_steps=None):
        num_sample_steps = default(num_sample_steps, 50)
        shape = (batch_size, 3, 256, 256)
        sigmas = self.sample_schedule(num_sample_steps)
        images = sigmas[0] * torch.randn(shape, device=self.device)
        
        for sigma in tqdm(sigmas):
            noise = torch.randn_like(images)
            noised_images = images + sigma * noise
            model_output = self.forward(noised_images, sigma)
            denoised = noised_images - sigma * model_output
            images = denoised
        
        return unnormalize_to_zero_to_one(images)
    
    def forward(self, images, sigma):
        batch_size, c, h, w, device = *images.shape, images.device
        assert h == 256 and w == 256, f'height and width of image must be 256'
        assert c == 3, 'mismatch of image channels'
        
        images = normalize_to_neg_one_to_one(images)
        noise = torch.randn_like(images)
        noised_images = images + sigma * noise
        
        denoised = self.network(noised_images, sigma)
        
        losses = F.mse_loss(denoised, images, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * (sigma ** 2)
        
        return losses.mean()
    
    def network(self, images, sigma):
        # Placeholder for the actual model
        pass

# Example usage:
diffusion = GaussianDiffusion(sigma_min=1e-4, sigma_max=20., device='cpu')
sampled_images = diffusion.sample(batch_size=16)
print(sampled_images.shape)  # Should print torch.Size([16, 3, 256, 256])