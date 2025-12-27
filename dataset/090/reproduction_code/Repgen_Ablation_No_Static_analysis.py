import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys

class EncoderWithUnusedParams(nn.Module):
    def __init__(self, dim=256, num_tokens=1000, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])
        # This is the problematic parameter - it exists but isn't used in forward
        self.to_logits = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        # Note: to_logits is not used here
        return x.mean(dim=1)  # Just pooling, not using to_logits

class ClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderWithUnusedParams()
        # Another classifier that's actually used
        self.classifier = nn.Linear(256, 10)
    
    def forward(self, x):
        features = self.encoder(x)
        # Using classifier instead of encoder.to_logits
        return self.classifier(features)

def static_analysis():
    """Perform static analysis to detect unused parameters"""
    model = ClassifierModel()
    
    # Get all parameters
    all_params = dict(model.named_parameters())
    print(f"Total parameters: {len(all_params)}")
    
    # Simulate a forward pass to track used parameters
    input_data = torch.randint(0, 1000, (2, 20))
    
    # Set up hooks to track parameters
    used_params = set()
    hooks = []
    
    def hook_fn(name):
        def hook(grad):
            used_params.add(name)
        return hook
    
    # Register hooks for all parameters
    for name, param in model.named_parameters():
        hooks.append(param.register_hook(hook_fn(name)))
    
    # Forward and backward pass
    output = model(input_data)
    dummy_loss = output.sum()
    dummy_loss.backward()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Check which parameters weren't used
    unused_params = set(all_params.keys()) - used_params
    print("\nUnused parameters:")
    for name in unused_params:
        print(f"- {name}")
    
    # Specifically check the parameter of interest
    problem_param = 'encoder.to_logits.weight'
    if problem_param in unused_params:
        print(f"\n✓ Bug confirmed: '{problem_param}' is not used during forward/backward pass")
        print("This will cause issues with DDP, which requires all parameters to be used")
    else:
        print(f"\n✗ Bug not found: '{problem_param}' appears to be used in computation")

if __name__ == "__main__":
    static_analysis()