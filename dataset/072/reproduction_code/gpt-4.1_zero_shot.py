import torch
import deepspeed
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
ds_config = {
    "train_batch_size": 1,
    "fp16": {"enabled": False}
}

model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)

print(model_engine.model)