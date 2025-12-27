import torch

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, dmodel: int, nhead: int):
        super().__init__()
        self.register_buffer('bias', None)
        
# Create an instance of the model without initializing biases
model = RelPositionMultiHeadedAttention(32, 4)

# Print the current state_dict to show uninitialized bias parameters
print("State dict before initialization:")
print(model.state_dict())

# Modify the bias parameter to be initialized with zeros
for name, param in model.named_parameters():
    if 'bias' in name:
        print(f"Initializing {name}:")
        param.data = torch.zeros_like(param)

model.eval()

# Print the updated state_dict after initializing biases
print("\nState dict after initializing biases to zero:")
print(model.state_dict())

# Example forward pass (assuming X is a valid input)
X = torch.randn(1, 32)  # Arbitrary input for demonstration

try:
    output = model(X)
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Error during inference: {e}")

import torch

class RelPositionMultiHeadedAttention(nn.Module):
    def __init__(self, dmodel: int, nhead: int):
        super().__init__()
        self.register_buffer('bias', None)
        
# Create an instance of the model without initializing biases
model = RelPositionMultiHeadedAttention(32, 4)

print("State dict before initialization:")
print(model.state_dict())

# Initialize parameters to zero instead of using torch.Tensor
for name, param in model.named_parameters():
    if 'bias' in name:
        param.data = torch.zeros_like(param)

print("\nState dict after initializing biases to zero:")
print(model.state_dict())