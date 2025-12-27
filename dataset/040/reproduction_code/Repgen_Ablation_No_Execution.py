import torch
from torchvision import datasets, transforms
from labml_nn.diffusion.ddpm.unet import UNet
from labml import labml, runner

batch_size = 32
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

unet = UNet()
learning_rate = 1e-3
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler()

criterion = torch.nn.CrossEntropyLoss()

x = torch.randn(batch_size, 3, 32, 32)
output = unet(x)
target = torch.randint(0, 10, (batch_size,))
loss = criterion(output, target)

labml.monitor.add('loss', loss.item())
assert not torch.isnan(loss).item(), "Loss is NaN"