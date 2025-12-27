import torch
from vector_quantize_pytorch import ResidualVQ

def check_initialized_mlps(model):
    has_mlps = any('implicit_neural_codebook' in name and param.numel() > 0
                   for name, param in model.named_parameters())
    return has_mlps

# Create ResidualVQ with implicit_neural_codebook=False
vq = ResidualVQ(dim=64, codebook_size=1024, num_quantizers=4, implicit_neural_codebook=False)

# Check if MLPs were initialized anyway
has_mlps = check_initialized_mlps(vq)

print(f"MLPs initialized despite implicit_neural_codebook=False: {has_mlps}")
print("Model parameters:", [name for name, _ in vq.named_parameters()])