import torch
from x_transformers import ContinuousTransformerWrapper, Encoder

def minimal_reproduction():
    """
    Minimal code to reproduce the bug with ContinuousTransformerWrapper return_mems
    """
    # Create a small model for quick testing
    model = ContinuousTransformerWrapper(
        dim = 64,
        max_seq_len = 128,
        attn_layers = Encoder(
            dim = 64,
            depth = 3
        )
    )
    
    # Input data
    x = torch.randn(1, 128, 64)
    
    # Create memory tokens (3 layers of dimension 64)
    mems = [torch.randn(1, 32, 64) for _ in range(3)]
    
    # Case 1: With mems=None (works correctly)
    output1 = model(x, mems=None, return_mems=True)
    
    if isinstance(output1, tuple) and len(output1) == 2:
        logits1, mems1 = output1
        print("Case 1 (mems=None):")
        print(f"  ✓ Returned tuple with logits {logits1.shape} and {len(mems1)} memory tensors")
    else:
        print("Case 1 (mems=None):")
        print(f"  ✗ Unexpected return type: {type(output1)}")
    
    # Case 2: With non-None mems (bug case)
    output2 = model(x, mems=mems, return_mems=True)
    
    if isinstance(output2, tuple) and len(output2) == 2:
        logits2, mems2 = output2
        print("\nCase 2 (with mems):")
        print(f"  Returned tuple with logits {logits2.shape}")
        
        if mems2 is None:
            print(f"  ✓ Bug reproduced: mems is None")
        elif len(mems2) == 0:
            print(f"  ✓ Bug reproduced: mems is an empty list")
        elif len(mems2) != 3:
            print(f"  ✓ Bug reproduced: mems has {len(mems2)} tensors, expected 3")
        else:
            print(f"  ✗ Failed to reproduce bug: mems contains {len(mems2)} tensors as expected")
    else:
        print("\nCase 2 (with mems):")
        print(f"  ✓ Bug reproduced: Unexpected return type {type(output2)}")
    
    # Case 3: With non-None mems but return_mems=False
    output3 = model(x, mems=mems, return_mems=False)
    
    print("\nCase 3 (with mems, return_mems=False):")
    if isinstance(output3, torch.Tensor):
        print(f"  ✓ Correctly returned only logits with shape {output3.shape}")
    else:
        print(f"  ✗ Unexpected return type: {type(output3)}")

if __name__ == "__main__":
    minimal_reproduction()