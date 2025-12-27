import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class AttentionConfig:
    dim: int = 512
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    qk_norm: bool = False
    kv_heads: int = None

class PlanAttention(nn.Module):
    """
    Planning implementation to identify the bug with qk_norm and kv_heads
    """
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        self.heads = config.heads
        self.kv_heads = config.kv_heads or config.heads
        self.dim_head = config.dim_head
        
        inner_dim = config.dim_head * config.heads
        inner_dim_kv = config.dim_head * self.kv_heads
        
        # Initialize QKV projections
        self.to_q = nn.Linear(config.dim, inner_dim, bias=False)
        self.to_k = nn.Linear(config.dim, inner_dim_kv, bias=False)
        self.to_v = nn.Linear(config.dim, inner_dim_kv, bias=False)
        
        # QK normalization parameters
        self.qk_norm = config.qk_norm
        if config.qk_norm:
            # Plan: Show both incorrect and correct implementations
            print(f"\nPlanning QK norm scales:")
            print(f"Heads: {config.heads}, KV Heads: {self.kv_heads}")
            
            # Incorrect implementation (the bug)
            print("\nIncorrect implementation (bug):")
            print(f"qk_norm_q_scale: shape = ({config.heads}, 1, {config.dim_head})")
            print(f"qk_norm_k_scale: shape = ({config.heads}, 1, {config.dim_head})")
            
            # Correct implementation
            print("\nCorrect implementation (fix):")
            print(f"qk_norm_q_scale: shape = ({config.heads}, 1, {config.dim_head})")
            print(f"qk_norm_k_scale: shape = ({self.kv_heads}, 1, {config.dim_head})")
            
            # Create the parameters (using the buggy implementation for demonstration)
            self.qk_norm_q_scale = nn.Parameter(torch.ones(config.heads, 1, config.dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(config.heads, 1, config.dim_head))  # BUG HERE
            
            # Optional: Create a fixed version for comparison
            self.fixed_qk_norm_k_scale = nn.Parameter(torch.ones(self.kv_heads, 1, config.dim_head))
    
    def check_shapes(self, x):
        """Check the shapes of the tensors during computation"""
        b, n, d = x.shape
        
        # Compute q, k, v
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        # Reshape
        q = q.view(b, n, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, n, self.kv_heads, self.dim_head).transpose(1, 2)
        v = v.view(b, n, self.kv_heads, self.dim_head).transpose(1, 2)
        
        print(f"\nShape analysis:")
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        
        if self.qk_norm:
            print(f"qk_norm_q_scale shape: {self.qk_norm_q_scale.shape}")
            print(f"qk_norm_k_scale shape: {self.qk_norm_k_scale.shape}")
            
            if self.kv_heads != self.heads:
                print("\n✗ BUG IDENTIFIED: qk_norm_k_scale has incorrect shape!")
                print(f"Expected: ({self.kv_heads}, 1, {self.dim_head})")
                print(f"Actual: ({self.heads}, 1, {self.dim_head})")
                
                # Show what happens with the buggy implementation
                print("\nSimulating the bug:")
                try:
                    k_norm = F.normalize(k, dim=-1)
                    k_scaled = k_norm * self.qk_norm_k_scale
                    print("This should fail but might not in this simplified version")
                except Exception as e:
                    print(f"Error as expected: {e}")
                
                # Show the fix
                print("\nSimulating the fix:")
                k_norm = F.normalize(k, dim=-1)
                k_scaled = k_norm * self.fixed_qk_norm_k_scale
                print(f"Fixed version works with k_scaled shape: {k_scaled.shape}")
            else:
                print("\n✓ No issue detected (heads == kv_heads)")

def run_plan():
    # Test different configurations
    configs = [
        AttentionConfig(qk_norm=True, kv_heads=8, heads=8),  # Should be fine
        AttentionConfig(qk_norm=True, kv_heads=2, heads=8),  # Should detect bug
        AttentionConfig(qk_norm=False, kv_heads=2, heads=8)  # No qk_norm, should be fine
    ]
    
    for i, config in enumerate(configs):
        print(f"\n{'='*40}")
        print(f"Configuration {i+1}:")
        print(f"heads={config.heads}, kv_heads={config.kv_heads}, qk_norm={config.qk_norm}")
        
        model = PlanAttention(config)
        x = torch.randn(2, 16, config.dim)
        model.check_shapes(x)

if __name__ == "__main__":
    run_plan()