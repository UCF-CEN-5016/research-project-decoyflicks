import torch

# Define batch size and input dimensions
batch_size = 32
max_seq_len = 1024
feature_dim = 768

# Create random uniform input data x with shape (batch_size, max_seq_len, feature_dim)
x = torch.rand((batch_size, max_seq_len, feature_dim))

# Initialize cos_cached and sin_cached tensors with shapes (max_seq_len, 4) where feature_dim is the original feature dimension
cos_cached = torch.randn(max_seq_len, 4)
sin_cached = torch.randn(max_seq_len, 4)

# Simulate a scenario where only a partial set of features are processed by setting a variable `d` to 3
d = 3

# Create neg_half_x tensor with shape (batch_size, max_seq_len, d) filled with negative half values
neg_half_x = torch.full((batch_size, max_seq_len, d), -0.5)

# Call the problematic statement: x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])
try:
    x_rope = (x * cos_cached[:x.shape[0]]) + (neg_half_x * sin_cached[:x.shape[0]])
except RuntimeError as e:
    print(e)

# Assert that a `RuntimeError` occurs with the message 'The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 3'
assert str(e) == 'The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 3'

# Monitor the shape of cos_cached and sin_cached during execution to verify they are not resized correctly
print(cos_cached.shape)
print(sin_cached.shape)