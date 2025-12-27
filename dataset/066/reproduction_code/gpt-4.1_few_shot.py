import torch
import torch.nn as nn

# Dummy selective_kernel_feature_fusion implementation
def selective_kernel_feature_fusion(x1, x2, x3):
    # Just sum the inputs as a placeholder
    return x1 + x2 + x3

# Simulated feature maps (e.g. outputs from DAUs)
level1_dau_2 = torch.randn(1, 64, 32, 32)
level2_dau_2 = torch.randn(1, 64, 32, 32)  # Unused variable in original code
level3_dau_2 = torch.randn(1, 64, 32, 32)

# Incorrect fusion: level2_dau_2 is not used, level3_dau_2 used twice
skff_incorrect = selective_kernel_feature_fusion(level1_dau_2, level3_dau_2, level3_dau_2)

# Correct fusion: use level2_dau_2 as intended
skff_correct = selective_kernel_feature_fusion(level1_dau_2, level2_dau_2, level3_dau_2)

print("Incorrect fusion sum:", skff_incorrect.sum().item())
print("Correct fusion sum:", skff_correct.sum().item())