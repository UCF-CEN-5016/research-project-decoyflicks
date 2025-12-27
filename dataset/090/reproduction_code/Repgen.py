import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

class CustomEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, num_classes=10):
        super().__init__()
        # Model components
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=512), 
            num_layers=2
        )
        # This parameter is problematic - not used in forward pass
        self.to_logits = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Embedding and transformer processing
        x = self.embed(x)
        x = self.transformer(x)
        # Just return the mean of embeddings
        # Note: to_logits is not used here!
        return x.mean(dim=1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CustomEncoder()
        # Using this classifier instead of encoder.to_logits
        self.classifier = nn.Linear(256, 10)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

def print_param_status(model, step=""):
    """Print gradient status of parameters"""
    print(f"\n--- Parameter Status {step} ---")
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"{name}: No gradient")
        else:
            print(f"{name}: Has gradient (norm: {param.grad.norm().item():.6f})")

def initialize_distributed():
    """Initialize the distributed environment"""
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    # Get rank and world size from environment or use defaults
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend='gloo',  # Using gloo for CPU testing
        rank=rank,
        world_size=world_size
    )
    
    print(f"Initialized process {rank} of {world_size}")
    return rank, world_size

def train_step(model, inputs, targets, optimizer):
    """Perform one training step"""
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    
    # Return loss before optimizer step
    return loss.item()

def reproduce_bug_all_steps():
    # Step 1: Set up distributed environment
    try:
        rank, world_size = initialize_distributed()
    except Exception as e:
        print(f"Distributed initialization failed: {e}")
        print("Running in non-distributed mode for demonstration")
        rank, world_size = 0, 1
    
    # Step 2: Create model
    model = Model()
    
    # Step 3: Wrap with DDP if in distributed mode
    if world_size > 1:
        ddp_model = DDP(model, find_unused_parameters=False)  # This will expose the bug
    else:
        ddp_model = model  # Just use the model directly for testing
    
    # Step 4: Create optimizer
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    
    # Step 5: Generate dummy data
    inputs = torch.randint(0, 1000, (8, 20))
    targets = torch.randint(0, 10, (8,))
    
    # Step 6: Initial parameter check
    if rank == 0:
        print_param_status(model, "Before Training")
        
        # Specifically check the problematic parameter
        problem_param = model.encoder.to_logits.weight
        print(f"\nProblem parameter shape: {problem_param.shape}")
        print(f"Initially zero: {(problem_param == problem_param.new_zeros(problem_param.shape)).all()}")
    
    # Step 7: Training loop
    try:
        for epoch in range(3):
            loss = train_step(ddp_model, inputs, targets, optimizer)
            
            if rank == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
                
                # Check parameters after backward
                if epoch == 1:
                    print_param_status(model, f"After Epoch {epoch+1} Backward")
            
            # This should raise an error in DDP mode with find_unused_parameters=False
            optimizer.step()
            
        # Step 8: Final parameter check
        if rank == 0:
            print_param_status(model, "After Training")
            
            # Verify the problem parameter specifically
            problem_param = model.encoder.to_logits.weight
            print(f"\nProblem parameter 'encoder.to_logits.weight':")
            print(f"Has gradient: {problem_param.grad is not None}")
            if problem_param.grad is not None:
                print(f"Gradient norm: {problem_param.grad.norm().item():.6f}")
            
            print("\nBug status: ", end="")
            if problem_param.grad is None:
                print("✓ Bug reproduced - parameter has no gradient")
            else:
                print("✗ Bug not reproduced - parameter has gradient")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("This error likely indicates the DDP bug with unused parameters")
        print("DDP requires all parameters to participate in the forward pass")

if __name__ == "__main__":
    reproduce_bug_all_steps()