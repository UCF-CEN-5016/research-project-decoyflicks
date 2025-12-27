import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class QKVAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        qk_norm = False,
        kv_heads = None
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.kv_heads = kv_heads or heads
        
        # Check for potential parameter conflict
        if qk_norm and self.kv_heads != heads:
            print(f"WARNING: Using qk_norm=True with kv_heads ({self.kv_heads}) != heads ({heads}) may cause errors")
        
        inner_dim = dim_head * heads
        inner_dim_kv = dim_head * self.kv_heads
        
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim_kv, bias = False)
        self.to_v = nn.Linear(dim, inner_dim_kv, bias = False)
        
        self.qk_norm = qk_norm
        if qk_norm:
            # Incorrect initialization - potential bug
            self.q_norm = nn.LayerNorm(dim_head)
            self.k_norm = nn.LayerNorm(dim_head)
            
            # The bug: q_scale uses heads but k_scale also uses heads instead of kv_heads
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))  # BUG HERE
            
            # Print the parameter shapes for analysis
            print(f"qk_norm_q_scale shape: {self.qk_norm_q_scale.shape}")
            print(f"qk_norm_k_scale shape: {self.qk_norm_k_scale.shape}")
            print(f"Expected k_scale shape for kv_heads: {(self.kv_heads, 1, dim_head)}")

    def static_analysis(self):
        # Perform static analysis to detect the bug
        if self.qk_norm and self.kv_heads != self.heads:
            q_shape = (1, self.heads, 10, self.dim_head)
            k_shape = (1, self.kv_heads, 10, self.dim_head)
            
            print(f"\nStatic analysis of tensor shapes:")
            print(f"q tensor shape: {q_shape}")
            print(f"k tensor shape: {k_shape}")
            print(f"qk_norm_q_scale shape: {(self.heads, 1, self.dim_head)}")
            print(f"qk_norm_k_scale shape: {(self.heads, 1, self.dim_head)}")
            
            # This analysis shows the bug: k has kv_heads but qk_norm_k_scale has heads
            if self.heads != self.kv_heads:
                print("\n✗ BUG DETECTED: Parameter shape mismatch!")
                print(f"k tensor has {self.kv_heads} heads but qk_norm_k_scale has {self.heads} heads")
                print("This will cause a runtime error when applying qk_norm_k_scale to k")
            else:
                print("\n✓ No shape mismatch detected")

def test_static_analysis():
    # Create instances with different configurations
    print("\nCase 1: qk_norm=True, kv_heads=heads (should be fine)")
    model1 = QKVAttention(dim=256, heads=8, kv_heads=8, qk_norm=True)
    model1.static_analysis()
    
    print("\nCase 2: qk_norm=True, kv_heads!=heads (should detect the bug)")
    model2 = QKVAttention(dim=256, heads=8, kv_heads=2, qk_norm=True)
    model2.static_analysis()
    
    print("\nCase 3: qk_norm=False, kv_heads!=heads (should be fine)")
    model3 = QKVAttention(dim=256, heads=8, kv_heads=2, qk_norm=False)
    model3.static_analysis()

if __name__ == "__main__":
    test_static_analysis()