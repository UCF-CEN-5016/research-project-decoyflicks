import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

# Minimal setup
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.to_logits = nn.Linear(10, 10)  # Example layer

    def forward(self, x):
        # Simulate the forward pass without using to_logits.weight
        return x  # Note: This is a simplification and not typical usage

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.decoder_layer(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Setup DDP
def setup_ddp(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

# Main function to reproduce the bug
def main():
    rank = 0  # Replace with appropriate rank for multi-GPU setup
    world_size = 1  # For single GPU, set to 1
    setup_ddp(rank, world_size)

    model = Model()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # Dummy data
    inputs = torch.randn(1, 10).to(device)
    labels = torch.randn(1, 10).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # Check if encoder.to_logits.weight is updated
        print("Encoder to logits weight:", ddp_model.module.encoder.to_logits.weight)

if __name__ == "__main__":
    main()