import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc(x)
        return x

def main():
    model = Model()

    engine, _, _, _ = deepspeed.initialize(args=deepspeed.InitArgs(local_rank=0), model=model)

    x = torch.randn(1, 5)
    output = engine(x)

    handle = deepspeed.comm.all_reduce(output)

if __name__ == "__main__":
    main()