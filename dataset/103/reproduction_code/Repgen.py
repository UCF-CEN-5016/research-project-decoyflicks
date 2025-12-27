import torch
from vector_quantize_pytorch import ResidualLFQ

def reproduce_bug():
    """
    Reproduce the commit loss calculation issue with mask and ResidualLFQ
    """
    print("Setting up the ResidualLFQ model with masked input...")
    
    # Model parameters
    dim = 14               # Input dimension 
    codebook_size = 16     # Small codebook for faster testing
    num_quantizers = 3     # Multiple quantizers as in ResidualLFQ
    
    # Create the model with commitment loss enabled
    model = ResidualLFQ(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        commitment_loss_weight=1.0,  # Important: set to > 0 to trigger the bug
    )
    
    # Create a batch of data with sequence-like shape
    batch_size = 2
    seq_len = 1851
    
    # Create the input tensor
    x = torch.randn(batch_size, seq_len, dim)
    
    # Create a mask where some tokens are masked out
    # This is crucial for reproducing the bug
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    mask[:, 800:] = False  # Mask out part of the sequence
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}, Masked tokens: {(~mask).sum().item()}")
    
    # Forward pass with the mask - this should trigger the warning/error
    print("\nPerforming forward pass with mask (should show warning)...")
    try:
        with torch.autograd.detect_anomaly():
            quantized, indices, commit_loss = model(x, mask=mask)
        
        print(f"Forward pass completed - quantized shape: {quantized.shape}")
        print(f"Commit loss shape: {commit_loss.shape}")
        print(f"Commit loss values: {commit_loss}")
        
        # Check for warning in the output
        print("\nCheck the output for the following warning:")
        print("UserWarning: Using a target size that is different to the input size.")
        print("This will likely lead to incorrect results due to broadcasting.")
        
    except RuntimeError as e:
        if "shape mismatch" in str(e).lower() or "size mismatch" in str(e).lower():
            print(f"✓ Bug reproduced: {e}")
        else:
            print(f"Different runtime error: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    reproduce_bug()