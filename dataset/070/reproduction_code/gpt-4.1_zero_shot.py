import torch
import deepspeed
import torch.nn as nn

deepspeed.init_distributed()

class DummyLayer(nn.Module):
    def forward(self, x):
        # This will trigger deepspeed.comm.all_reduce call internally in some models
        # Here we call it explicitly to reproduce the error
        deepspeed.comm.all_reduce(x)
        return x

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = DummyLayer()
    def forward(self, x):
        return self.layer(x)

model = DummyModel()
model = deepspeed.initialize(model=model, config={"train_micro_batch_size_per_gpu":1})[0]

input_tensor = torch.randn(2).cuda()
output = model(input_tensor)