import torch

def reproduce_bug():
    print("Setting up the SinusoidalPosEmb class with missing theta initialization...")
    
    class SinusoidalPosEmb:
        def __init__(self, dim):
            self.dim = dim
            # Bug: theta is not initialized here
            
        def forward(self, x):
            # This will raise a NameError because theta is not defined
            return x * theta  # Bug: using theta instead of self.theta
    
    # Create an instance
    dim = 32
    emb = SinusoidalPosEmb(dim)
    
    # Try to use the forward method
    x = torch.randn(10)
    
    try:
        print("Calling forward method (should fail with NameError)...")
        result = emb.forward(x)
        print("✗ Bug not reproduced - forward method executed without error")
        return False
    except NameError as e:
        if "name 'theta' is not defined" in str(e):
            print(f"✓ Bug reproduced: {e}")
            return True
        else:
            print(f"Different NameError: {e}")
            return False
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    reproduce_bug()