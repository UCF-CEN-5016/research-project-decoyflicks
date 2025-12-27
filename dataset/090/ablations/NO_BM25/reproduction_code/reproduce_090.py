import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_logits = nn.Linear(10, 5)

    def forward(self, x):
        return self.to_logits(x)

class Decoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)

def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = Encoder().to(device)
    decoder = Decoder(encoder).to(device)
    
    encoder = nn.parallel.DistributedDataParallel(encoder)
    decoder = nn.parallel.DistributedDataParallel(decoder)

    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    input_tensor = torch.randn(8, 10).to(device)
    target_tensor = torch.randint(0, 5, (8,)).to(device)

    for epoch in range(5):
        optimizer.zero_grad()
        output = decoder(input_tensor)
        loss = nn.functional.cross_entropy(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch}, encoder.to_logits.weight: {encoder.module.to_logits.weight.data}')

if __name__ == "__main__":
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)