import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import os

class Encoder(nn.Module):
    def __init__(self, dim=512, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(1000, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8), 
            num_layers=4
        )
        # Here's the problematic parameter - not directly used in forward
        self.to_logits = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # Normal forward pass
        x = self.embedding(x)
        x = self.encoder(x)
        
        # Note that we don't use self.to_logits here
        # This is the bug - the parameter exists but isn't used
        return x.mean(dim=1)  # Just return the embeddings

class Classifier(nn.Module):
    def __init__(self, encoder_dim=512, num_classes=10):
        super().__init__()
        self.encoder = Encoder(encoder_dim, num_classes)
        # Using a separate classifier instead of encoder.to_logits
        self.classifier = nn.Linear(encoder_dim, num_classes)
        
    def forward(self, x):
        # Get encoded representations
        encoded = self.encoder(x)
        # Use the classifier instead of encoder.to_logits
        logits = self.classifier(encoded)
        return logits

def setup_distributed():
    # Initialize the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    torch.distributed.init_process_group(
        backend='gloo',  # Using gloo for CPU-based testing
        rank=rank,
        world_size=world_size
    )
    return rank

def reproduce_bug():
    # Set up distributed environment
    rank = setup_distributed()
    device = torch.device(f"cpu")  # Using CPU for simplicity
    
    # Create model and move to device
    model = Classifier().to(device)
    ddp_model = DDP(model)
    
    # Check parameter status before training
    print(f"Before training - to_logits.weight.grad: {model.encoder.to_logits.weight.grad}")
    
    # Create dummy data
    inputs = torch.randint(0, 1000, (8, 32)).to(device)
    targets = torch.randint(0, 10, (8,)).to(device)
    
    # Training loop
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    for i in range(3):
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        # Check if all gradients are computed
        if i == 2:
            # Check which parameters have gradients
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Parameter {name} has no gradient")
                else:
                    print(f"Parameter {name} has gradient with norm {param.grad.norm().item()}")
        
        optimizer.step()
    
    # Specifically check the problematic parameter
    print(f"After training - to_logits.weight.grad: {model.encoder.to_logits.weight.grad}")
    print(f"Is to_logits.weight updated: {model.encoder.to_logits.weight.grad is not None}")

if __name__ == "__main__":
    reproduce_bug()