import torch
from torchvision import datasets, transforms
from tqdm import trange
from torch.utils.data import DataLoader
import torch.nn.functional as F

torch.manual_seed(1234)

batch_size = 256
height, width = 28, 28
channels = 1

input_data = torch.rand(batch_size, height, width, channels)

# Assuming SimpleFSQAutoEncoder is defined elsewhere in the codebase or imported from a module
model = SimpleFSQAutoEncoder(levels=[8, 6, 5], implicit_neural_codebook=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
input_data = input_data.to(device)

for epoch in trange(1000):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output, _, _ = model(batch)
        loss = F.mse_loss(output, batch)
        loss.backward()
        optimizer.step()

# Verify MLP initialization
mlp_params = [param for name, param in model.named_parameters() if 'mlp' in name]
assert all(param.requires_grad for param in mlp_params), "MLP parameters not initialized correctly"

# Monitor memory usage
import psutil
memory_usage = psutil.Process().memory_info().rss / 1024**2

print(f"Memory Usage: {memory_usage} MB")

# Assert MLP parameters still contain weights and biases
assert all('weight' in param or 'bias' in param for param in mlp_params), "MLP parameters missing weights or biases"