import torch
import deepspeed
from deepspeed.runtime.comm.comm import init_distributed

# Simulate Domino initialization failure
def faulty_init():
    # This mimics the uninitialized cdb state
    return None

# Monkey-patch the init function to reproduce the bug
deepspeed.runtime.comm.comm.init_distributed = faulty_init

# Minimal Domino setup that triggers the error
model = torch.nn.Linear(10, 10)
engine, _, _, _ = deepspeed.initialize(
    model=model,
    config={
        "train_batch_size": 8,
        "domino_enabled": True,  # Enable Domino
        "domino_model_parallel_size": 1
    }
)

# This would trigger the AttributeError
output = engine(torch.randn(2, 10))
print(output)