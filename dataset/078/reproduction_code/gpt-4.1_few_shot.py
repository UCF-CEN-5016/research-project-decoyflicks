import torch
from torch.autograd import Function

# Minimal mockup of NestedTensor and unbind operation that causes backward shape mismatch

class DummyNestedTensor:
    def __init__(self, tensors):
        self.tensors = tensors

    def unbind(self):
        # Simulate unbind returning a tuple of tensors
        return tuple(self.tensors)

class UnbindBackwardAutogradNestedTensor0(Function):
    @staticmethod
    def forward(ctx, nested_tensor):
        ctx.save_for_backward(*nested_tensor.tensors)
        return nested_tensor.unbind()

    @staticmethod
    def backward(ctx, *grads):
        saved = ctx.saved_tensors
        # Intentionally return a gradient with wrong shape for index 0
        wrong_grad = torch.ones_like(saved[0]).view(5, 2)
        # Expected to return a tuple of gradients, but first one is incorrectly shaped
        return (wrong_grad,)

# Prepare dummy input with correct shape
x = torch.randn(5, 10, 1024, requires_grad=True)

# Construct dummy nested tensor (simulate batch of 5, with inner dims)
nested = DummyNestedTensor([x])

# Use the custom backward function that triggers the error
try:
    outputs = UnbindBackwardAutogradNestedTensor0.apply(nested)
    loss = sum(o.sum() for o in outputs)
    loss.backward()
except RuntimeError as e:
    print(f"Caught error during backward: {e}")