import torch
from torch import nn, optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# Set seed for reproducibility
torch.manual_seed(1234)

# Load dataset and create DataLoader
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Define the SimpleFSQAutoEncoder class
class SimpleFSQAutoEncoder(nn.Module):
    def __init__(self, levels, implicit_neural_codebook=False):
        super(SimpleFSQAutoEncoder, self).__init__()
        # ... (rest of the code as provided in the context)

    # ... (rest of the methods as provided in the context)

# Create an instance of the model
model = SimpleFSQAutoEncoder(levels=[8, 6, 5], implicit_neural_codebook=False).cuda()

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(1000):
    for data in dataloader:
        data = data.cuda()
        
        optimizer.zero_grad()
        quantize, embed_ind, loss = model(data)
        loss.backward()
        optimizer.step()

# Check if any MLPs were initialized
mlp_weights = [param for name, param in model.named_parameters() if 'mlp' in name and param.requires_grad]
assert all(param.sum().item() != 0 for param in mlp_weights), "MLPs were not initialized during training."