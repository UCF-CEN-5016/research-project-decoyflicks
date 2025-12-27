import torch
from vector_quantize_pytorch.sim_vq import ResidualSimVQ

def minimal_reproduction():
    """
    Minimal reproduction of the undefined return_loss bug in ResidualSimVQ
    """
    # Create a minimal ResidualSimVQ instance
    model = ResidualSimVQ(
        dim=32,              # Small dimension for faster execution
        num_quantizers=2,    # Minimum number needed for quantize_dropout
        codebook_size=64,    # Small codebook size
        quantize_dropout=True,
        channels_first=True  # This combination with dropout triggers shape issues
    )
    
    # Create a small input tensor
    x = torch.randn(1, 32, 10)  # (batch, channels, seq_len)
    
    # Print the relevant part of the forward method
    import inspect
    forward_source = inspect.getsource(ResidualSimVQ.forward)
    print("Forward method code snippet:")
    for line in forward_source.split('\n'):
        if 'return_loss' in line:
            print(f"BUG HERE >>> {line}")
        elif 'all_losses' in line and 'stack' in line:
            print(line)
    
    # Try to run the forward method
    try:
        result = model(x)
        print("\n✗ Bug not reproduced - no error occurred")
    except NameError as e:
        if "return_loss" in str(e):
            print(f"\n✓ Bug reproduced: {e}")
            print("\nThis occurs because the code references 'return_loss' which is never defined.")
            print("The fix is to add 'return_loss=True' as a parameter to the forward method.")
        else:
            print(f"\n✗ Different error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    minimal_reproduction()