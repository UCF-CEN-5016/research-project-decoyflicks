import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.to_logits = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    model = Encoder().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    x = torch.randn(20, 10).to(rank)
    y = torch.randn(20, 10).to(rank)
    
    for _ in range(5):
        optimizer.zero_grad()
        output = ddp_model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"to_logits.weight grad: {model.to_logits.weight.grad}")
    
    cleanup()

def run():
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    run()