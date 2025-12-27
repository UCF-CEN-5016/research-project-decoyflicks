import torch

# Check for MPS backend availability
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except AttributeError as e:
    print(f"Error: {e}")
    print("MPS backend not available in this PyTorch version")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define and move model to device
model = torch.nn.Linear(10, 5).to(device)

# Generate input tensor and move to device
x = torch.randn(3, 10).to(device)

# Perform forward pass on the model
output = model(x)
print(output)