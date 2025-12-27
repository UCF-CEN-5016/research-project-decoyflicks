import torch
from vector_quantize_pytorch import ResidualVQ

def demonstrate_bug():
    # Create ResidualVQ with implicit_neural_codebook=False (default)
    residual_vq = ResidualVQ(
        dim=64,
        codebook_size=512,
        num_quantizers=4
    )
    
    # Check if MLP parameters exist when they shouldn't
    has_mlps = any('mlp' in name for name, _ in residual_vq.named_parameters())
    num_params = sum(p.numel() for p in residual_vq.parameters())
    
    print(f"Has MLP parameters when implicit_neural_codebook=False: {has_mlps}")
    print(f"Total parameters: {num_params}")
    
    # Now create with implicit_neural_codebook=True for comparison
    residual_vq_with_mlp = ResidualVQ(
        dim=64,
        codebook_size=512,
        num_quantizers=4,
        implicit_neural_codebook=True
    )
    
    num_params_with_mlp = sum(p.numel() for p in residual_vq_with_mlp.parameters())
    print(f"\nTotal parameters with MLPs enabled: {num_params_with_mlp}")
    print(f"Parameter difference: {num_params_with_mlp - num_params} (should be 0 when implicit_neural_codebook=False)")

if __name__ == '__main__':
    demonstrate_bug()