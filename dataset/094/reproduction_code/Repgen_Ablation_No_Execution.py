import torch
from x_transformers import ContinuousTransformerWrapper, Encoder

def reproduce_bug():
    print("Setting up ContinuousTransformerWrapper...")
    
    # Define model parameters
    dim = 512
    depth = 6
    
    # Create the model
    model = ContinuousTransformerWrapper(
        dim = dim,
        max_seq_len = 1024,
        attn_layers = Encoder(
            dim = dim,
            depth = depth
        )
    )
    
    # Create input tensors
    x = torch.randn(1, 1024, dim)  # Batch size 1, sequence length 1024, dimension 512
    mask = torch.ones(1, 1024).bool()  # All tokens are valid
    
    # Create memory tokens
    mems = [torch.randn(1, 100, dim) for _ in range(depth)]
    
    print("Testing with mems=None (should work correctly)...")
    # Test with mems=None (should work correctly)
    output_none = model(x, mask=mask, mems=None, return_mems=True)
    
    if isinstance(output_none, tuple) and len(output_none) == 2:
        logits_none, mems_none = output_none
        print(f"✓ Success: With mems=None, returned both logits {logits_none.shape} and mems {len(mems_none)}")
    else:
        print(f"✗ Failed: With mems=None, did not return expected tuple, got {type(output_none)}")
    
    print("\nTesting with non-None mems (should fail)...")
    # Test with non-None mems (should fail)
    output_mems = model(x, mask=mask, mems=mems, return_mems=True)
    
    if isinstance(output_mems, tuple) and len(output_mems) == 2:
        logits_mems, mems_returned = output_mems
        if len(mems_returned) == depth:
            print(f"✗ Failed to reproduce bug: With non-None mems, correctly returned both logits and mems")
            for i, mem in enumerate(mems_returned):
                print(f"  Memory {i} shape: {mem.shape}")
        else:
            print(f"✓ Bug reproduced: With non-None mems, returned logits but invalid mems {len(mems_returned)}")
    else:
        print(f"✓ Bug reproduced: With non-None mems, did not return expected tuple, got {type(output_mems)}")

if __name__ == "__main__":
    reproduce_bug()