import torch
from torch import nn
from vit_pytorch.nest import register_nestingoma, UnbindNestingTensor
from vit_pytorch.nest_tensor_3d import (
    _NestedToken,
    _UnbindLeftRight,
    get_nesting_shape,
    UnbindNestingTensor3D
)

register_nest = register_nestingoma

def forward_backward(model, x):
    # Get the nesting shape for input tensor
    B, N, D = get_nesting_shape(x)
    
    # Initialize nested tensor with left and right branches
    left = nn.Parameter(torch.randn(B, 1))
    right = nn.Parameter(torch.randn(B, 1))
    
    outputs = {
        'left': left,
        'right': right
    }
    
    for j in range(N):
        if j > 0:
            # Split input into two branches: left and right parts
            s = x[:j]
            t = x[j:]
            
            output1, output2 = model(s, t)
            
            outputs['left'] += output1
            outputs['right'] += output2
    
    return outputs

if __name__ == "__main__":
    # Minimal code to test the bug
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    B = 5
    N = 10
    D = 1024
    
    x = nn.Parameter(torch.randn(B, N, D)).to(device)
    model = ... # Initialize your model here
    
    try:
        outputs = forward_backward(model, x)
        print("Bug fixed training possible\n", outputs)
    except Exception as e:
        print(f"Error: {e}")