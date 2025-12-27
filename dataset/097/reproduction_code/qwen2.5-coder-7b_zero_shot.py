import torch
from classifier_free_guidance import GaussianDiffusion

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gaussian_diffusion = GaussianDiffusion()
        
    def forward(self)
        self.gaussian_diffusion.offset_noise_strength()

model = Net()