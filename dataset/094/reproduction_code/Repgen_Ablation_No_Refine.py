import torch
from x_transformers import ContinuousTransformerWrapper, Encoder

def inspect_bug_with_breakpoints():
    """
    Reproduce the bug with debugging information at key points
    """
    print("Setting up the model...")
    
    # Model parameters
    dim = 512
    depth = 6
    batch_size = 1
    seq_len = 256  # Shorter sequence for faster execution
    mem_len = 100
    
    # Create model
    model = ContinuousTransformerWrapper(
        dim = dim,
        max_seq_len = seq_len,
        attn_layers = Encoder(
            dim = dim,
            depth = depth
        )
    )
    
    # Input data
    x = torch.randn(batch_size, seq_len, dim)
    mask = torch.ones(batch_size, seq_len).bool()
    
    # Memory tokens
    print("\nCreating memory tokens...")
    mems = [torch.randn(batch_size, mem_len, dim) for _ in range(depth)]
    
    # Helper function to check memory token stats
    def check_mems(mems_list, name):
        if mems_list is None:
            print(f"{name} is None")
            return
        
        print(f"{name} contains {len(mems_list)} tensors")
        if len(mems_list) > 0:
            first_mem = mems_list[0]
            print(f"  First memory tensor shape: {first_mem.shape}")
            print(f"  Memory tensor min/max/mean: {first_mem.min().item():.4f}/{first_mem.max().item():.4f}/{first_mem.mean().item():.4f}")
    
    # Original memory stats
    check_mems(mems, "Original mems")
    
    # Track the forward pass
    print("\nMonkey patching model to track internal state...")
    
    # Save original method
    original_forward = model.forward
    
    # Counter for tracking calls
    call_counter = 0
    
    # Define a tracking forward method
    def tracking_forward(self, x, mask = None, mems = None, return_mems = False, **kwargs):
        nonlocal call_counter
        call_counter += 1
        
        print(f"\nForward pass #{call_counter}")
        print(f"Input: x.shape = {x.shape}")
        print(f"Args: return_mems = {return_mems}, mems is {'provided' if mems is not None else 'None'}")
        
        if mems is not None:
            check_mems(mems, "Input mems")
        
        # Call original method
        result = original_forward(self, x, mask, mems, return_mems, **kwargs)
        
        if return_mems:
            if isinstance(result, tuple) and len(result) == 2:
                logits, new_mems = result
                print(f"Result: logits.shape = {logits.shape}")
                check_mems(new_mems, "Output mems")
            else:
                print(f"Result: Unexpected return type {type(result)}")
        else:
            print(f"Result: logits.shape = {result.shape}")
        
        return result
    
    # Apply monkey patch
    model.forward = tracking_forward.__get__(model, type(model))
    
    # Test case 1: mems=None
    print("\n\n=== Test Case 1: mems=None, return_mems=True ===")
    result1 = model(x, mask=mask, mems=None, return_mems=True)
    
    # Test case 2: With non-None mems
    print("\n\n=== Test Case 2: With mems provided, return_mems=True ===")
    result2 = model(x, mask=mask, mems=mems, return_mems=True)
    
    # Test case 3: Without return_mems
    print("\n\n=== Test Case 3: With mems provided, return_mems=False ===")
    result3 = model(x, mask=mask, mems=mems, return_mems=False)
    
    # Compare results
    print("\n\n=== Results Summary ===")
    print("Test Case 1 (mems=None, return_mems=True):")
    if isinstance(result1, tuple) and len(result1) == 2:
        print(f"✓ Successfully returned tuple with logits and mems")
    else:
        print(f"✗ Unexpected return type: {type(result1)}")
    
    print("\nTest Case 2 (mems provided, return_mems=True):")
    if isinstance(result2, tuple) and len(result2) == 2:
        logits2, mems2 = result2
        if mems2 is None or len(mems2) == 0 or len(mems2) != depth:
            print(f"✓ Bug reproduced: mems is {'None' if mems2 is None else f'list with {len(mems2)} items (expected {depth})'}")
        else:
            print(f"✗ Failed to reproduce bug: mems contains {len(mems2)} items as expected")
    else:
        print(f"✓ Bug reproduced: unexpected return type: {type(result2)}")
    
    print("\nTest Case 3 (mems provided, return_mems=False):")
    if isinstance(result3, torch.Tensor):
        print(f"✓ Correctly returned only logits")
    else:
        print(f"✗ Unexpected return type: {type(result3)}")
    
    # Restore original method
    model.forward = original_forward

if __name__ == "__main__":
    inspect_bug_with_breakpoints()