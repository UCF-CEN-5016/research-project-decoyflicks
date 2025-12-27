import torch

class Test:
    def __init__(self, d):
        self.d = d
        self.cos_cached = torch.randn(10, 1, 1, d)
        self.sin_cached = torch.randn(10, 1, 1, d)

    def forward(self, x):
        x_rope = x[:, :, :, :self.d]
        neg_half_x = -x[:, :, :, :self.d] / 2
        x_rope = (x_rope * self.cos_cached[:x.shape[0], :, :, :self.d]) + (neg_half_x * self.sin_cached[:x.shape[0], :, :, :self.d])
        return x_rope

test = Test(4)
x = torch.randn(3, 1, 1, 4)
print(test.forward(x))

class TestError:
    def __init__(self, d):
        self.d = d
        self.cos_cached = torch.randn(10, 1, 1, d)
        self.sin_cached = torch.randn(10, 1, 1, d)

    def forward(self, x):
        x_rope = x[:, :, :, :self.d]
        neg_half_x = -x[:, :, :, :self.d] / 2
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
        return x_rope

test_error = TestError(4)
x = torch.randn(3, 1, 1, 4)
try:
    print(test_error.forward(x))
except RuntimeError as e:
    print(e)