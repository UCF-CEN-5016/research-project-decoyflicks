import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * 128 * 128),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        z = z.view(-1, 3, 128, 128)
        return z

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 128 * 128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = img.view(-1, 3 * 128 * 128)
        validity = self.model(img)
        return validity

def main():
    parser = argparse.ArgumentParser(description='DCGAN for CelebA')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root='path_to_celeba', transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator()
    discriminator = Discriminator()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        for i, (imgs, _) in enumerate(dataloader):
            valid = torch.ones(imgs.size(0), 1)
            fake = torch.zeros(imgs.size(0), 1)

            z = torch.randn(imgs.size(0), 100)
            gen_imgs = generator(z)

            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

if __name__ == '__main__':
    main()