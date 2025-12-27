import labml
from labml import monit, tracker, experiment
import torch
from torch import nn
from labml_nn.optimizers import Adam, AMSGrad

# Define a batch size of 1 and sequence length of 20
batch_size = 1
sequence_length = 20
model_dim = 768

# Create a random input tensor x with shape (batch_size, sequence_length, model_dim) where model_dim = 768
x = torch.randn(batch_size, sequence_length, model_dim)

# Initialize cos_cached and sin_cached tensors with shapes (1, 1, model_dim // 2, sequence_length) filled with zeros
cos_cached = torch.zeros(1, 1, model_dim // 2, sequence_length)
sin_cached = torch.zeros(1, 1, model_dim // 2, sequence_length)

# Negate half of the input tensor x to create neg_half_x
neg_half_x = -x[:, :, :model_dim // 2]

# Assert that the dimensions of cos_cached and sin_cached are (1, 1, 384, 20)
assert cos_cached.shape == (1, 1, model_dim // 2, sequence_length)

# Attempt to compute x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]), where self is an instance of AMSGrad
with experiment.record(name='synthetic', comment='AMSGrad'):
    for step in monit.loop(1):
        x_rope = x[:, :, :model_dim // 2] * cos_cached[:batch_size, :, :, :sequence_length] + neg_half_x * sin_cached[:batch_size, :, :, :sequence_length]