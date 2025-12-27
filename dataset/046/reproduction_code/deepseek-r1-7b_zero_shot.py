import torch
from pytorch_audio import models, alignment

# Simulate model outputs for demonstration
batch_size = 1
input_length = 20
num_classes = 3

probs = torch.randn(batch_size, input_length, num_classes)  # Shape: (batch_size, input_length, num_classes)
log_probs = torch.log(probs)

# Ensure log_probs is reshaped to be at least 3D by adding a dimension if necessary
if len(log_probs.shape) < 3:
    log_probs = log_probs.unsqueeze(1)  # Adds a new axis at position 1

# Now, log_probs should have shape (batch_size, input_length, num_classes)
print(f"log_probs shape: {log_probs.shape}")