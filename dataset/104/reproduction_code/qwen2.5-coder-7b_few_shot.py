import torch
from vector_quantize_pytorch import ResidualVQ

def check_mlp_initialization(model):
    # Search through all submodules to detect any 'mlp' attributes (subtle change:
    # look inside nested codebooks where MLPs may be incorrectly initialized)
    mlp_locations = []
    for name, module in model.named_modules():
        if hasattr(module, 'mlp'):
            mlp_locations.append(name or '<root>')
    if mlp_locations:
        return f"MLP exists in modules: {mlp_locations}"
    return "MLP not found"

# Create a ResidualVQ instance with implicit_neural_codebook disabled
vq_model = ResidualVQ(num_embeddings=512, embedding_dim=64, num_codebooks=2, implicit_neural_codebook=False)

# Check if MLPs are initialized and print the result
mlp_initialization_status = check_mlp_initialization(vq_model)
print(mlp_initialization_status)