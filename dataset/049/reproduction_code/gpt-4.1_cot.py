import torch

# Create a tensor using torch.Tensor (uninitialized)
uninit_tensor = torch.Tensor(10, 10)
print("Uninitialized tensor (torch.Tensor(10,10)):")
print(uninit_tensor)
print("Sum:", uninit_tensor.sum())

# Create a tensor using torch.zeros (initialized to zeros)
zero_tensor = torch.zeros(10, 10)
print("\nZero initialized tensor (torch.zeros(10,10)):")
print(zero_tensor)
print("Sum:", zero_tensor.sum())

# Demonstrate that uninitialized tensor may have garbage or NaN values
nan_count = torch.isnan(uninit_tensor).sum().item()
print(f"\nNumber of NaNs in uninitialized tensor: {nan_count}")

# If using these as nn.Parameter for bias terms in a model, the uninitialized tensor can cause instability
bias_u = torch.nn.Parameter(torch.Tensor(10))  # uninitialized
bias_v = torch.nn.Parameter(torch.Tensor(10))  # uninitialized

print("\nBias parameters initialized with torch.Tensor (uninitialized):")
print("bias_u:", bias_u)
print("bias_v:", bias_v)
print("Sum bias_u:", bias_u.sum())
print("Sum bias_v:", bias_v.sum())

# Proper initialization with zeros
bias_u_fixed = torch.nn.Parameter(torch.zeros(10))
bias_v_fixed = torch.nn.Parameter(torch.zeros(10))

print("\nBias parameters initialized with torch.zeros (initialized):")
print("bias_u_fixed:", bias_u_fixed)
print("bias_v_fixed:", bias_v_fixed)
print("Sum bias_u_fixed:", bias_u_fixed.sum())
print("Sum bias_v_fixed:", bias_v_fixed.sum())