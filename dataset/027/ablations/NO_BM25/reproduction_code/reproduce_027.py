import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

def main():
    parser = argparse.ArgumentParser(description='DCGAN for CelebA')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(root='../data/CelebA', transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            labels = torch.full((batch_size,), 1, device=device)

            optimizerD.zero_grad()
            output = discriminator(real_images).view(-1)
            errD_real = criterion(output, labels)
            errD_real.backward()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = generator(noise)
            labels.fill_(0)
            output = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output, labels)
            errD_fake.backward()
            optimizerD.step()

            optimizerG.zero_grad()
            labels.fill_(1)
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, labels)
            errG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{args.epochs}] Batch [{i}/{len(train_loader)}] Loss D: {errD_real.item() + errD_fake.item()}, Loss G: {errG.item()}')

if __name__ == '__main__':
    main()