import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Define the core issue: dimension mismatch in BCE loss
def reproduce_dimension_mismatch():
    # Parameters matching the bug report
    batch_size = 64
    image_size = 128  # 128x128 images as mentioned in bug report
    
    # Create model outputs with shape that would cause the issue
    # The shape [batch_size, 50] represents the flattened discriminator output
    # 50 is chosen to make the total elements 3200 (matching error)
    model_output = torch.sigmoid(torch.randn(batch_size, 50))
    model_output_flattened = model_output.view(-1)  # Shape becomes [3200]
    
    # Create targets with incorrect dimension that would trigger the error
    # Shape [128] as mentioned in error message
    targets = torch.ones(128)
    
    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    
    # This should raise ValueError with nearly identical message to bug report
    try:
        loss = criterion(model_output_flattened, targets)
    except ValueError as e:
        print(f"Successfully reproduced bug: {e}")
        print(f"Target size: {targets.size()} vs Input size: {model_output_flattened.size()}")
        # Verify it matches the bug description
        assert "target size" in str(e).lower() and "input size" in str(e).lower()
        assert str(targets.size()) in str(e) and str(model_output_flattened.size()) in str(e)

if __name__ == "__main__":
    reproduce_dimension_mismatch()