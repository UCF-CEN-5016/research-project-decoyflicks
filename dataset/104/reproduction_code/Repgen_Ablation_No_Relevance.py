import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from vector_quantize_pytorch.residual_vq import SimpleVQAutoEncoder

# Define batch size and image dimensions
batch_size = 256
height, width = 28, 28
channels = 1

# Create random uniform input data
input_data = torch.randn(batch_size, height, width, channels).float().uniform_(-1, 1)

# Initialize SimpleVQAutoEncoder
num_codes = 256
model = SimpleVQAutoEncoder(num_codes=num_codes, implicit_neural_codebook=False)

# Set learning rate and define number of training iterations
lr = 3e-4
train_iter = 1000

# Define FashionMNIST dataset and DataLoader
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a simple training loop
def train(model, train_loader, lr, train_iterations):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(train_iterations):
        model.train()
        for images, _ in train_loader:
            optimizer.zero_grad()
            output = model(images)
            loss = output[2]  # Assuming loss is the third element of the output tuple
            loss.backward()
            optimizer.step()

# Call the train function
train(model, train_loader, lr, train_iter)

# Verify MLP initialization and commit_loss during training
for name, param in model.named_parameters():
    if 'mlp' in name:
        print(f"{name} initialized: {param.requires_grad}")
assert torch.isclose(model.quantize.commit_loss, 0.0)