import torch
import deepspeed
import deepspeed.comm as dist_comm
from torch import nn

# Minimal model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        # Attempt to use DeepSpeed's all_reduce communication primitive
        # without initializing the communication backend
        reduced = deepspeed.comm.all_reduce(x)
        return self.linear(reduced)

def main():
    # Model and input
    model = DummyModel()

    x = torch.randn(2, 10).cuda()

    # Intentionally NOT calling deepspeed.init_distributed() or torch.distributed.init_process_group()
    # which means deepspeed.comm._get_cdb() returns None internally

    try:
        output = model(x)
        print("Output:", output)
    except AttributeError as e:
        print("Caught AttributeError:", e)

if __name__ == "__main__":
    main()