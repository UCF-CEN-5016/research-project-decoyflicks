import os
from random import Random
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Initialize process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    
    return torch.device('cuda', rank)

def worker(rank):
    # Setup and CUDA device
    if not (torch.cuda.is_available() and torch.cuda.device_count() > 1):
        print(f"Rank {rank} - CUDA unavailable or single GPU detected")
        return

    device = setup(rank, world_size=2)
    model = torch.nn.Sequential(
        torch.nn.Linear(32,64), # Input layer
        torch.nn.ReLU(),       # Hidden layer activation
        torch.nn.Linear(64,10)  # Output layer
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    
    dataset = MyDataset()  # Implement your dataset
    
    data_loader = DataLoader(dataset, batch_size=32,
                             shuffle=True,
                             drop_last=True)
    
    criterion = torch.nn.MSELoss()
    
    for epoch in range(2):
        model.train()
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            
            # Forward and backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Rank {rank} - Epoch {epoch}: Loss = {loss.item():.4f}")

def MyDataset():
    random.seed(42)
    dataset_size = 3000
    data = []
    
    # Scale across workers for proper data sharding
    if rank == 0:
        data += [(torch.randn(1, 32), torch.randn(1)) for _ in range(1500)]
    elif rank == 1:
        data += [(torch.randn(1, 32), torch.randn(1)) for _ in range(1500)]
    
    return data

if __name__ == '__main__':
    world_size = 2
    processes = []
    
    # Start worker on each rank
    for rank in range(world_size):
        p = os.path.join(os.environ.get(' slurm_node_id ',''), str(rank))
        if not os.path.exists(p):
            print(f"Skipping worker {rank} due to node path issue")
        else:
            print(f"\nStarting worker {rank} on device: {setup(rank, world_size=world_size)}")
            p = os.fork()
            if p > 0:
                worker(rank)
    
    # Wait for all workers
    for rank in range(world_size):
        processes.append(os.waitpid(0, os.W_CHILD))
    os.exit(0)