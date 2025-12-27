import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

def main():
    parser = argparse.ArgumentParser(description='DCGAN with CelebA dataset')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root='path_to_celeba', transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = Generator().to('cuda')
    discriminator = Discriminator().to('cuda')
    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        for i, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to('cuda')
            batch_size = real_images.size(0)

            noise = torch.randn(batch_size, 100, 1, 1).to('cuda')
            fake_images = generator(noise)

            real_labels = torch.ones(batch_size, 1).to('cuda')
            fake_labels = torch.zeros(batch_size, 1).to('cuda')

            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake

            print(outputs.shape)  # Should print (batch_size, 1)

            try:
                d_loss.backward()
            except ValueError as e:
                print(e)

if __name__ == '__main__':
    main()