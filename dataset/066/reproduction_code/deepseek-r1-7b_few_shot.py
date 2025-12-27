import numpy as np
import torch

# Create dummy data to simulate MIRNet layers
level1_data = torch.randn(32, 64)  # Level1 feature map
level2_data = torch.randn(32, 128)  # Level2 feature map (double the size)
level3_data = torch.randn(32, 256)  # Unnecessary level as per original code

# Incorrect usage of level3_data which shouldn't be used here
try:
    skff_ = selective_kernel_feature_fusion(level1_data, level3_data, level3_data)
except Exception as e:
    print(f"Error: {e}")  # This will show the error when using incorrect variable

# Correct usage with intended level2_data
skff_correct = selective_kernel_feature_fusion(level1_data, level2_data, level3_data)
print("Correct feature fusion output:", skff_correct)