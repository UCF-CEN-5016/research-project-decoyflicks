import torch
from torch import nn, F
from torch.nn.functional import mse_loss, logsigmoid
import math

# Importing missing functions and constants from the provided code snippet
def default(value, fallback):
    return value if value is not None else fallback()

def extract(a, t, x_shape):
    b, _, *spatial = x_shape
    out = a.gather(-1, t)
    return out.view(b, *([1] * len(spatial)), *spatial)

def unnormalize_to_zero_to_one(x):
    return 0.5 * (x + 1.)

def approx_standard_normal_cdf(x):
    c = torch.tensor([[-3.9625104e-08, 2.2072247e-07, -2.7886807e-06, 1.8628806e-05,
                        -8.604636e-05, 2.0548941e-04, -2.938034e-03, 2.6781408e-02,
                        -1.5203292e-01, 6.480541e-01, 1.0507561e+01, -1.8587378e+01,
                        1.4293448e+01, -4.5661995e+00]])
    a = torch.tensor([-8.784894e-01, 2.552881e-01])
    b = torch.tensor([8.827837e-01, -1.7064417e+00])
    c = torch.tensor([3.1383f, -1.1488f, 9.15f, -1.894f])
    d = torch.tensor([-8.422295f, 2.0706775f])
    p = ((x > a[0]) * (x < b[0])).float()
    q = ((x >= b[0]) & (x < c[0])).float()
    r = ((x >= c[0]) & (x < d[0])).float()

    return (p * (c[2] + p * c[3]) / (d[1] + p * d[2]) +
            q * (-a[0] + b[0] + (c[1] - a[1]) * q /
                  (b[1] - a[1]) + (c[3] - a[3]) * q / (d[1] - a[1])) +
            r * (-a[2] + b[2] + (c[4] - a[4]) * r /
                  (b[2] - a[2]) + (c[5] - a[5]) * r / (d[2] - a[2])))

# Rest of the code remains unchanged
class GaussianDiffusion:
    # ... (rest of the class definition)

class ModelPrediction:
    # ... (rest of the class definition)

class LearnedGaussianDiffusion(GaussianDiffusion):
    # ... (rest of the class definition)