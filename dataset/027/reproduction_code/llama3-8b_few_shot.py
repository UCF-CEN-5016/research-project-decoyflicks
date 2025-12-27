import torch
from torchvision import datasets, transforms
from DC_GAN.model import Generator, Discriminator

# Define the DCGAN model
G = Generator(100).to('cuda')  # Generator network
D = Discriminator().to('cuda')  # Discriminator network

# Define the dataset and data loader for CelebA
transform = transforms.Compose([transforms.CenterCrop(128), transforms.ToTensor()])
dataset = datasets.CelebA(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop that produces the error
for epoch in range(10):
    for i, (x, _) in enumerate(dataloader):
        # Perform training iteration
        z = torch.randn(x.shape[0], 100).to('cuda')
        output = G(z)
        errD_real = D(output)
        print(f"Epoch {epoch}, Iteration {i}: Error = {errD_real.item()}")