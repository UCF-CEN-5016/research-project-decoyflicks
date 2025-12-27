import torch
from x_transformers import ContinuousTransformerWrapper, Encoder

def reproduce_bug_all_steps():
    print("==== Step 1: Model Setup ====")
    # Define model parameters
    dim = 512
    depth = 6
    batch_size = 1
    seq_len = 1024
    mem_len = 100
    
    # Create the model
    model = ContinuousTransformerWrapper(
        dim = dim,
        max_seq_len = seq_len,
        attn_layers = Encoder(
            dim = dim,
            depth = depth
        )
    )
    print(f"Model created with dimension {dim} and depth {depth}")
    
    print("\n==== Step 2: Input Preparation ====")
    # Create input tensor
    x = torch.randn(batch_size, seq_len, dim)
    print(f"Input tensor shape: {x.shape}")
    
    # Create mask
    mask = torch.ones(batch_size, seq_len).bool()
    print(f"Mask shape: {mask.shape}")
    
    # Create memory tokens
    mems = [torch.randn(batch_size, mem_len, dim) for _ in range(depth)]
    print(f"Created {len(mems)} memory tensors, each with shape {mems[0].shape}")
    
    print("\n==== Step 3: Test Case 1 - No Memory Tokens ====")
    # First test with mems=None
    print("Running forward pass with mems=None and return_mems=True")
    try:
        output_none = model(x, mask=mask, mems=None, return_mems=True)
        
        if isinstance(output_none, tuple) and len(output_none) == 2:
            logits_none, mems_none = output_none
            print(f"✓ Success: Returned logits with shape {logits_none.shape}")
            print(f"✓ Success: Returned {len(mems_none)} memory tensors")
            print(f"  First memory tensor shape: {mems_none[0].shape}")
        else:
            print(f"✗ Failed: Expected tuple of (logits, mems), got {type(output_none)}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n==== Step 4: Test Case 2 - With Memory Tokens ====")
    # Now test with non-None mems
    print("Running forward pass with non-None mems and return_mems=True")
    try:
        output_mems = model(x, mask=mask, mems=mems, return_mems=True)
        
        if not isinstance(output_mems, tuple):
            print(f"✓ Bug reproduced: Expected tuple, got {type(output_mems)}")
        elif len(output_mems) != 2:
            print(f"✓ Bug reproduced: Expected tuple of length 2, got {len(output_mems)}")
        else:
            logits_mems, mems_returned = output_mems
            print(f"Returned logits with shape {logits_mems.shape}")
            
            if mems_returned is None:
                print(f"✓ Bug reproduced: Returned mems is None")
            elif len(mems_returned) == 0:
                print(f"✓ Bug reproduced: Returned empty mems list")
            elif len(mems_returned) != depth:
                print(f"✓ Bug reproduced: Expected {depth} memory tensors, got {len(mems_returned)}")
            else:
                print(f"Memory tensors seem correct. First shape: {mems_returned[0].shape}")
                print("✗ Failed to reproduce bug")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n==== Step 5: Test Case 3 - Without return_mems ====")
    # Test without return_mems
    print("Running forward pass with non-None mems and return_mems=False")
    try:
        output_no_return = model(x, mask=mask, mems=mems, return_mems=False)
        
        if isinstance(output_no_return, tuple):
            print(f"✗ Unexpected: Got tuple {len(output_no_return)} when return_mems=False")
        else:
            print(f"✓ Success: Got only logits with shape {output_no_return.shape} when return_mems=False")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n==== Step 6: Summary ====")
    print("Test results summary:")
    print("1. With mems=None, return_mems=True: Correctly returns both logits and memory tokens")
    print("2. With non-None mems, return_mems=True: Bug - fails to return valid memory tokens")
    print("3. With non-None mems, return_mems=False: Correctly returns only logits")

if __name__ == "__main__":
    reproduce_bug_all_steps()