import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class DCGANGenerator(nn.Module):
    def __init__(self):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
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
        return self.main(input)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch DCGAN Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CelebA(root='./data', transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = DCGANGenerator().to(device)
    discriminator = DCGANDiscriminator().to(device)

    criterion = nn.BCELoss()

    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        for i, data in enumerate(train_loader, 0):
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)

            # Train Discriminator
            optimizerD.zero_grad()
            label_real = torch.ones((batch_size, 1)).to(device)
            label_fake = torch.zeros((batch_size, 1)).to(device)

            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label_real)
            D_x = output.mean().item()

            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake = generator(noise)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label_fake)
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            errD.backward(retain_graph=True)
            optimizerD.step()

            # Train Generator
            optimizerG.zero_grad()
            output = discriminator(fake).view(-1)
            errG = criterion(output, label_real)
            D_G_z2 = output.mean().item()
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

if __name__ == '__main__':
    main()