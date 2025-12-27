import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simplified Self-Attention module similar to one in UNet
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channels = in_channels
        self.mha_head = nn.MultiheadAttention(in_channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([in_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x):
        batch_size, c, h, w = x.shape
        
        # Reshape for attention: [batch, channels, h*w] -> [batch, h*w, channels]
        x_reshaped = x.reshape(batch_size, c, h * w).transpose(1, 2)
        
        # Apply layer norm
        x_ln = self.ln(x_reshaped)
        
        # Problematic attention implementation (original version)
        q = x_ln
        k = x_ln
        v = x_ln
        
        # Bug demonstration: Computing attention scores with matrix multiplication
        # and applying softmax along incorrect dimension
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        
        # PROBLEMATIC LINE: Using dim=1 for softmax (incorrect)
        attn_wrong = F.softmax(attn_scores, dim=1)
        
        # Correct implementation: softmax along dim=2
        attn_correct = F.softmax(attn_scores, dim=2)
        
        # Compute outputs using both incorrect and correct attention
        out_wrong = torch.bmm(attn_wrong, v)
        out_correct = torch.bmm(attn_correct, v)
        
        # Reshape back
        out_wrong = out_wrong.transpose(1, 2).reshape(batch_size, c, h, w)
        out_correct = out_correct.transpose(1, 2).reshape(batch_size, c, h, w)
        
        return out_wrong, out_correct, attn_wrong, attn_correct

# Set up test parameters
batch_size = 2
channels = 16
height = 8
width = 8

# Create input tensor
x = torch.randn(batch_size, channels, height, width)

# Create self-attention module
sa = SelfAttention(channels)

# Forward pass
out_wrong, out_correct, attn_wrong, attn_correct = sa(x)

# Demonstrate the difference in attention distributions
print(f"Shape of attention matrices: {attn_wrong.shape}")

# Check that probabilities sum to 1 along the appropriate dimension
sum_wrong = attn_wrong.sum(dim=1)
sum_correct = attn_correct.sum(dim=2)

print("\nIncorrect Implementation (softmax dim=1):")
print(f"Sum along dim 1: {sum_wrong[0, 0:5]}")  # Should NOT be 1
print(f"Sum along dim 2: {attn_wrong.sum(dim=2)[0, 0:5]}")  # Should be inconsistent

print("\nCorrect Implementation (softmax dim=2):")
print(f"Sum along dim 1: {attn_correct.sum(dim=1)[0, 0:5]}")  # Should be inconsistent
print(f"Sum along dim 2: {sum_correct[0, 0:5]}")  # Should be 1.0

# Demonstrate impact on output representations
diff = (out_wrong - out_correct).abs().mean()
print(f"\nMean absolute difference in outputs: {diff.item()}")

# Check if the outputs are significantly different
significant_diff = diff > 0.01
print(f"Outputs are significantly different: {significant_diff}")