import deepspeed
import torch
import torch.nn as nn

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()

# Initialize DeepSpeed engine
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config={
        "train_batch_size": 1,
        "fp16": {"enabled": False}
    }
)

# Attempt to access model.model (incorrect, causes AttributeError)
try:
    print(model_engine.model)
except AttributeError as e:
    print(f"Caught error: {e}")

# Correct access is via model_engine.module
print("Accessing underlying model correctly:", model_engine.module)