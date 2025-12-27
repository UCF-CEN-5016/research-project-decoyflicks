import torch
from torch.nn.functional import softmax

# Create sample data representing attention scores with shape [batch_size=2, seq_len=3]
attn = torch.randn(2, 3)

def apply_softmax_correctly(attn):
    # Incorrect way: using the last dimension (dim=-1) which is same as dim=1 for a 2D tensor
    # This would cause incorrect behavior if intended to sum over another axis
    soft_attn = softmax(attn, dim=-1)
    return soft_attn

# Correct way: assuming we want to apply along the last dimension (dim=-1) which is valid
soft_attn_correct = apply_softmax_correctly(attn)

print("Incorrect application of softmax (should be same as correct in this case):")
print(soft_attn_correct)

# Example of how incorrect dim could lead to issues:
# Assuming 'attn' has shape [batch_size, heads, seq_len, seq_len]
# Correct approach is to apply along last dimension (dim=-1 or 3)
# Incorrect approach if intended to sum per head would be wrong

def incorrect_application(attn):
    soft_attn = F.softmax(attn, dim=-1)  # Correct for this case
    return soft_attn

# If the correct dim was different, say 2 (for another structure), changing it fixes the bug:
correct_dim = 2
soft_attn_corrected = F.softmax(attn, dim=correct_dim)