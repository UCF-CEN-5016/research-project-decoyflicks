import torch

# Initialize u_bias and v_bias using torch.Tensor which creates uninitialized values
u = torch.Tensor(1, 1)
v = torch.Tensor(100, 100)

print(u.sum())           # Outputs: tensor(0.)
print(v.sum())           # Outputs: tensor(4.9312e-38)
print((u + v).sum())     # Outputs: tensor(nan)