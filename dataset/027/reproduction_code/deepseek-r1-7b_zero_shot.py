class Discriminator(nn.Module):
    def __init__(self, input_dim=64):
        super(Discriminator, self).__init__()
        self.c1 = nn.ConvTranspose2d(input_dim, 64, kernel_size=5, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.c2 = nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.c3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2)
    
    def forward(self, x):
        x = self.relu1(self.batch_norm1(self.c1(x)))
        x = self.relu2(self.batch_norm2(self.c2(x)))
        x = self.c3(x).squeeze()
        return torch.sigmoid(x)