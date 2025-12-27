# This script is a minimal reproduction of the error. It requires a Slurm cluster with specific nodes and hostfile setup.
# The error arises due to incorrect hostfile configuration or DeepSpeed initialization issues.

import torch
import deepspeed

# Example training loop that triggers the error
model = torch.nn.Linear(10, 10).to('cuda')
optimizer = torch.optim.Adam(model.parameters())
data = torch.randn(10, 10).to('cuda')

with deepspeed.zero.Init():
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()