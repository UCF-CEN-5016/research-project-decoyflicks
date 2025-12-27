import torch
import inspect
from x_transformers import ContinuousTransformerWrapper, Encoder

def analyze_source_code():
    """
    Analyze the source code of ContinuousTransformerWrapper to identify
    the bug in return_mems functionality
    """
    print("Analyzing ContinuousTransformerWrapper.forward method...")
    
    # Get the source code of the forward method
    try:
        forward_source = inspect.getsource(ContinuousTransformerWrapper.forward)
        
        print("\nExcerpt from forward method:")
        print("-" * 60)
        
        # Find relevant lines related to mems and return_mems
        for line in forward_source.split('\n'):
            if 'mems' in line or 'return_mems' in line or 'return' in line:
                print(line)
        
        print("-" * 60)
        
    except Exception as e:
        print(f"Could not get source code: {e}")
    
    print("\nAnalyzing behavior through tests...")
    
    # Create a minimal model for testing
    dim = 64  # smaller for faster testing
    depth = 3
    model = ContinuousTransformerWrapper(
        dim = dim,
        max_seq_len = 128,
        attn_layers = Encoder(
            dim = dim,
            depth = depth
        )
    )
    
    # Input tensor
    x = torch.randn(1, 128, dim)
    mask = torch.ones(1, 128).bool()
    
    # Test cases
    print("\nTest case results:")
    
    # Test 1: No mems, no return_mems
    result1 = model(x, mask=mask)
    print(f"1. No mems, no return_mems: {type(result1)}")
    if isinstance(result1, torch.Tensor):
        print(f"   Output shape: {result1.shape}")
    
    # Test 2: No mems, with return_mems
    result2 = model(x, mask=mask, return_mems=True)
    print(f"2. No mems, with return_mems: {type(result2)}")
    if isinstance(result2, tuple):
        print(f"   Tuple length: {len(result2)}")
        if len(result2) == 2:
            logits, mems = result2
            print(f"   Logits shape: {logits.shape}")
            print(f"   Mems: {type(mems)}")
            if mems is not None:
                print(f"   Number of memory tensors: {len(mems)}")
                if len(mems) > 0:
                    print(f"   First memory tensor shape: {mems[0].shape}")
    
    # Test 3: With mems, no return_mems
    mems = [torch.randn(1, 100, dim) for _ in range(depth)]
    result3 = model(x, mask=mask, mems=mems)
    print(f"3. With mems, no return_mems: {type(result3)}")
    if isinstance(result3, torch.Tensor):
        print(f"   Output shape: {result3.shape}")
    
    # Test 4: With mems, with return_mems (should reproduce bug)
    result4 = model(x, mask=mask, mems=mems, return_mems=True)
    print(f"4. With mems, with return_mems: {type(result4)}")
    bug_reproduced = False
    
    if isinstance(result4, tuple):
        print(f"   Tuple length: {len(result4)}")
        if len(result4) == 2:
            logits, new_mems = result4
            print(f"   Logits shape: {logits.shape}")
            print(f"   Mems: {type(new_mems)}")
            
            if new_mems is None:
                bug_reproduced = True
                print("   ✓ Bug reproduced: mems is None")
            elif len(new_mems) == 0:
                bug_reproduced = True
                print("   ✓ Bug reproduced: mems is an empty list")
            elif len(new_mems) != depth:
                bug_reproduced = True
                print(f"   ✓ Bug reproduced: mems contains {len(new_mems)} tensors, expected {depth}")
            else:
                print(f"   Number of memory tensors: {len(new_mems)}")
                print(f"   First memory tensor shape: {new_mems[0].shape}")
    else:
        bug_reproduced = True
        print("   ✓ Bug reproduced: did not return a tuple")
    
    # Conclusion
    print("\nAnalysis conclusion:")
    if bug_reproduced:
        print("✓ Bug successfully reproduced:")
        print("  When both mems is provided and return_mems=True, the function fails to")
        print("  correctly return the updated memory tokens.")
        print("\nPossible causes:")
        print("1. The code might be overwriting or not updating the mems variable")
        print("2. There might be a conditional branch that handles the return differently")
        print("   when mems is not None")
        print("3. The memory updating logic may have a bug when starting with pre-existing mems")
    else:
        print("✗ Failed to reproduce the bug in this environment")

if __name__ == "__main__":
    analyze_source_code()