import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class Encoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.proj = nn.Linear(32, dim)
        self.to_logits = nn.Linear(dim, 10)  # The problematic layer
        
    def forward(self, x):
        x = self.proj(x)  # to_logits.weight never used here
        return x  # But it's still in parameters()

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to rank
    encoder = Encoder().to(rank)
    ddp_encoder = DDP(encoder, device_ids=[rank])
    
    # Verify parameters
    print(f"Rank {rank} parameters before:")
    for name, param in ddp_encoder.named_parameters():
        print(f"{name}: {param.requires_grad}, {param.grad is None}")
    
    # Dummy training step
    optimizer = torch.optim.Adam(ddp_encoder.parameters())
    x = torch.randn(4, 32).to(rank)
    y = torch.randn(4, 128).to(rank)
    
    loss = (ddp_encoder(x) - y).pow(2).mean()
    loss.backward()
    optimizer.step()
    
    # Check gradients after step
    print(f"\nRank {rank} parameters after:")
    for name, param in ddp_encoder.named_parameters():
        print(f"{name}: grad exists? {param.grad is not None}")
    
    cleanup()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)