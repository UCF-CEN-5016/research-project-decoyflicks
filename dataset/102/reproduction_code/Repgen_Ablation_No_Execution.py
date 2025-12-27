import torch
from vector_quantize_pytorch import ResidualSimVQ

def reproduce_bug():
    print("Creating ResidualSimVQ model with quantize_dropout=True...")
    
    # Create model with the conditions that trigger the bug
    model = ResidualSimVQ(
        dim=64,                     # Embedding dimension
        num_quantizers=3,           # Number of quantizers
        codebook_size=256,          # Size of each codebook
        quantize_dropout=True,      # Enable quantize dropout (needed for bug)
        channels_first=True         # Use channels_first format (contributes to shape issues)
    )
    
    # Create input tensor with channels_first format (batch, channels, seq_len)
    x = torch.randn(2, 64, 16)
    
    print("Calling forward method (this should trigger the NameError)...")
    
    try:
        # This should raise NameError for undefined 'return_loss'
        output, indices, losses = model(x)
        print("✗ Bug not reproduced - no error occurred")
    except NameError as e:
        if "return_loss" in str(e):
            print(f"✓ Bug reproduced: {e}")
            
            # Show the location of the bug
            print("\nThe bug is in the ResidualSimVQ.forward method:")
            print("if return_loss:  # This variable is not defined")
            print("    all_losses = torch.stack(all_losses, dim=-1)")
            
            # Show the fix
            print("\nPossible fix:")
            print("# Add a parameter to the forward method")
            print("def forward(self, x, mask=None, return_loss=True, return_all_codes=False, ...)")
        else:
            print(f"Different error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    reproduce_bug()