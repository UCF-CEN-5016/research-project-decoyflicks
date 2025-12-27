import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from functools import partial

def exists(val):
    return val is not None

def round_up_multiple(n, m):
    return ((n + m - 1) // m) * m

# Simple implementation of SimVQ to avoid dependencies
class SimVQ(nn.Module):
    def __init__(self, dim, codebook_size, commitment_weight=1.0):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        
        # Initialize codebook
        self.register_buffer('codebook', torch.randn(codebook_size, dim))
    
    def forward(self, x, mask=None):
        # Simple distance calculation
        dist = torch.cdist(x.reshape(-1, self.dim), self.codebook)
        indices = torch.argmin(dist, dim=-1).reshape(x.shape[:-1])
        
        # Get codebook vectors
        indices_flat = indices.reshape(-1)
        quantized = self.codebook[indices_flat].reshape(x.shape)
        
        # Commitment loss
        commit_loss = F.mse_loss(x, quantized.detach()) 
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, indices, commit_loss * self.commitment_weight

# This is a simplified version of the actual ResidualSimVQ with the bug
class ResidualSimVQ(nn.Module):
    def __init__(
        self,
        dim,
        num_quantizers,
        codebook_size,
        commitment_weight=1.0,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        channels_first=False,
        accept_image_fmap=False
    ):
        super().__init__()
        self.dim = dim
        self.num_quantizers = num_quantizers
        self.channels_first = channels_first
        self.accept_image_fmap = accept_image_fmap
        
        # Create multiple quantizer layers
        self.layers = nn.ModuleList([
            SimVQ(
                dim=dim,
                codebook_size=codebook_size,
                commitment_weight=commitment_weight
            )
            for _ in range(num_quantizers)
        ])
        
        # Quantize dropout parameters
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of
    
    def forward(self, x, mask=None, return_all_codes=False, rand_quantize_dropout_fixed_seed=None):
        # Handle channels_first format
        if self.channels_first and not self.accept_image_fmap:
            x = x.transpose(1, 2)
        
        num_quant, quant_dropout_multiple_of, device = self.num_quantizers, self.quantize_dropout_multiple_of, x.device
        
        quantized_out = 0.
        residual = x
        
        all_losses = []
        all_indices = []
        
        should_quantize_dropout = self.training and self.quantize_dropout
        
        # Sample a layer index at which to dropout further residual quantization
        if should_quantize_dropout:
            # Set random seed for reproducibility
            if rand_quantize_dropout_fixed_seed is None:
                rand_quantize_dropout_fixed_seed = random.randint(0, 10000)
            
            random.seed(rand_quantize_dropout_fixed_seed)
            
            # Random index for dropout
            rand_quantize_dropout_index = random.randint(
                self.quantize_dropout_cutoff_index, 
                num_quant - 1
            )
            
            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = round_up_multiple(rand_quantize_dropout_index + 1, quant_dropout_multiple_of) - 1
            
            # Create null tensors
            if self.accept_image_fmap:
                null_indices = torch.full((*x.shape[:-1], 1), -1, device=device)
            else:
                null_indices = torch.full((*x.shape[:-1],), -1, device=device)
                
            null_loss = torch.tensor(0., device=device)
        
        # Go through the layers
        for quantizer_index, layer in enumerate(self.layers):
            # Skip layers after dropout index
            if should_quantize_dropout and quantizer_index > rand_quantize_dropout_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue
            
            # Apply quantization
            quantized, indices, loss = layer(residual, mask=mask)
            
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            
            all_indices.append(indices)
            all_losses.append(loss)
        
        # THIS IS THE BUG - return_loss is not defined
        if return_loss:  # NameError: return_loss is not defined
            all_losses = torch.stack(all_losses, dim=-1)
        
        # Handle channels_first format for output
        if self.channels_first and not self.accept_image_fmap:
            quantized_out = quantized_out.transpose(1, 2)
        
        if not return_all_codes:
            return quantized_out, all_indices, all_losses
        
        # If we need to return all codes
        all_codes = []
        for i, layer in enumerate(self.layers):
            codes = layer.codebook[all_indices[i]]
            all_codes.append(codes)
        
        return quantized_out, all_indices, all_losses, all_codes

def reproduce_bug_with_full_implementation():
    """
    Full reproduction of the bug with a complete implementation of ResidualSimVQ
    """
    print("Testing ResidualSimVQ with the 'return_loss' bug...")
    
    # Create model with necessary parameters
    model = ResidualSimVQ(
        dim=64,
        num_quantizers=3,
        codebook_size=256,
        quantize_dropout=True,
        channels_first=True
    )
    
    # Create input tensor (batch, channels, seq_len)
    x = torch.randn(2, 64, 16)
    
    try:
        # This should trigger the NameError
        output, indices, losses = model(x)
        print("✗ Bug not reproduced - no error occurred")
    except NameError as e:
        if "return_loss" in str(e):
            print(f"✓ Bug reproduced: {e}")
            
            print("\nThe bug occurs in the forward method of ResidualSimVQ:")
            print("  if return_loss:  # Variable is not defined anywhere")
            print("      all_losses = torch.stack(all_losses, dim=-1)")
            
            print("\nHow to fix:")
            print("1. Add 'return_loss=True' as a parameter to the forward method:")
            print("   def forward(self, x, mask=None, return_loss=True, return_all_codes=False, ...):")
            
            print("\n2. The second issue with shape inconsistencies happens when:")
            print("   - quantize_dropout=True and channels_first=True")
            print("   - the null_indices shape doesn't match other indices when transposing")
        else:
            print(f"Different error: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    reproduce_bug_with_full_implementation()