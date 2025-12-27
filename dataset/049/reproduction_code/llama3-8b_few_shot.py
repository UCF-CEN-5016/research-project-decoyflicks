import torch
import torch.nn as nn

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, num_heads):
        super(RelPositionMultiHeadedAttention, self).__init__()
        self.bias = torch.Tensor(1, 1)  # Uninitialised bias tensor

    def forward(self, x):
        # ... (rest of the implementation)

# Minimal reproduction example
attention_layer = RelPositionMultiHeadedAttention(num_heads=8)
print("Initialising attention layer...")
result = attention_layer(torch.randn(1, 10))
print("Result:", result)