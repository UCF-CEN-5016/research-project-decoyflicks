import torch
import inspect

def reproduce_bug_all_steps():
    print("=== Step 1: Original Class with Bug ===")
    
    # Define the buggy class
    class SinusoidalPosEmb:
        def __init__(self, dim):
            self.dim = dim
            # Bug: theta is not initialized
            
        def forward(self, x):
            # Bug: Using theta without initializing it
            half_dim = self.dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(-emb * torch.arange(half_dim))
            emb = x[:, None] * emb[None, :]
            emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
            # The bug is here - theta is used but not defined
            return emb * theta  # This will cause a NameError
    
    # Show the class definition
    print("SinusoidalPosEmb class definition:")
    print(inspect.getsource(SinusoidalPosEmb))
    
    print("\n=== Step 2: Testing the Buggy Class ===")
    
    # Create an instance
    emb = SinusoidalPosEmb(dim=32)
    
    # Check the instance attributes
    print("Instance attributes:")
    for attr in dir(emb):
        if not attr.startswith('__'):
            print(f"  {attr}: {getattr(emb, attr)}")
    
    # Create a tensor
    x = torch.linspace(0, 1, 10)
    print(f"\nInput tensor: {x}")
    
    # Try to call forward
    print("\nAttempting to call forward method...")
    try:
        result = emb.forward(x)
        print("✗ No error - bug not reproduced")
    except NameError as e:
        print(f"✓ NameError as expected: {e}")
        if "name 'theta' is not defined" in str(e):
            print("  This confirms the bug: theta is referenced but not defined")
        else:
            print("  Different NameError than expected")
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
    
    print("\n=== Step 3: Fixed Version ===")
    
    # Define the fixed class
    class FixedSinusoidalPosEmb:
        def __init__(self, dim):
            self.dim = dim
            # Fix: Initialize theta
            self.theta = torch.tensor(1.0)
            
        def forward(self, x):
            half_dim = self.dim // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(-emb * torch.arange(half_dim))
            emb = x[:, None] * emb[None, :]
            emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
            # Fix: Use self.theta instead of theta
            return emb * self.theta
    
    # Show the fixed class definition
    print("FixedSinusoidalPosEmb class definition:")
    print(inspect.getsource(FixedSinusoidalPosEmb))
    
    # Test the fixed class
    print("\nTesting fixed class...")
    fixed_emb = FixedSinusoidalPosEmb(dim=32)
    
    try:
        result = fixed_emb.forward(x)
        print(f"✓ Fixed version works correctly")
        print(f"  Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ Error in fixed version: {type(e).__name__}: {e}")
    
    print("\n=== Step 4: Summary ===")
    print("Bug: The 'theta' variable is referenced in the forward method but never defined.")
    print("Fix: Initialize 'self.theta' in the __init__ method and use 'self.theta' in forward.")
    print("This is a common mistake when converting from functional to object-oriented code,")
    print("where a variable is used but not properly made into an instance attribute.")

if __name__ == "__main__":
    reproduce_bug_all_steps()