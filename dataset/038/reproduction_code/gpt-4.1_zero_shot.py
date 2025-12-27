import torch

class RotPosEmb:
    def __init__(self):
        self.cos_cached = torch.ones(2, 1, 1, 4)
        self.sin_cached = torch.ones(2, 1, 1, 4)
        self.d = 3

    def apply(self, x):
        neg_half_x = -x[..., 1::2],  # dummy operation to match shape partially
        x_rope = x
        # buggy line causing size mismatch
        return (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x[0] * self.sin_cached[:x.shape[0]])

x = torch.randn(2, 1, 1, 3)
rpe = RotPosEmb()
rpe.apply(x)