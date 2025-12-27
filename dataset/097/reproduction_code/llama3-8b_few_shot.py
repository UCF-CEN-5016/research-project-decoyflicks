import os
from classifier_free_guidance import GaussianDiffusion

# Initialize the diffusion model with missing device attribute
model = GaussianDiffusion()

try:
    # Attempt to access the missing attribute
    print(model.device)
except AttributeError as e:
    print(f"Error: {e}")