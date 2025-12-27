import torch
import deepspeed
import os
from torch import nn
from deepspeed.runtime.engine import DeepSpeedEngine

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def forward(self, x):
        x = self.layer(x)
        deepspeed.comm.all_reduce(x, async_op=True)
        return x

def initialize_distributed():
    deepspeed.init_distributed(dist_backend='nccl')

def main():
    initialize_distributed()
    
    model = SimpleModel()
    engine = DeepSpeedEngine(
        model=model,
        config={
            "train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 0}
        },
        model_parameters=model.parameters(),
        optimizer=None,
        training_data=None
    )
    
    input = torch.randn(2, 10).cuda()
    output = engine(input)
    print(output)

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    main()