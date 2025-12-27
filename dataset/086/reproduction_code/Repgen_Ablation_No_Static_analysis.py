import torch
from einops import rearrange

def isolate_rotary_xpos_error():
    # Recreate the exact scenario from the error message
    # The power tensor has shape [1, 32] according to the error
    power = torch.randn(1, 32)
    scale = torch.tensor(0.9)
    
    print(f"Power shape: {power.shape}")
    
    try:
        # This is the line causing the error in the original code
        result = scale ** rearrange(power, 'n -> n 1')
        print("Transformation successful (unexpected)")
    except Exception as e:
        print(f"Error: {e}")
        print("Successfully reproduced the exact error")
    
    # Show the correct way to handle this shape
    try:
        # Option 1: Fix by correctly specifying the input dimensions
        result = scale ** rearrange(power, 'b n -> b n 1')
        print("Fixed with correct dimensions specification")
    except Exception as e:
        print(f"Fix 1 failed: {e}")
    
    try:
        # Option 2: Fix by flattening first
        power_flat = power.reshape(-1)
        result = scale ** rearrange(power_flat, 'n -> n 1')
        print("Fixed with flattening approach")
    except Exception as e:
        print(f"Fix 2 failed: {e}")

isolate_rotary_xpos_error()