import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# Set up a minimal environment
class DALLE(nn.Module):
    def __init__(self):
        super(DALLE, self).__init__()
        self.fc1 = nn.Linear(128, 128)  # dummy layer

    def forward(self, text, image_codes):
        x = self.fc1(text)
        return x

class TrainData(Dataset):
    def __init__(self, text, image_codes, mask):
        self.text = text
        self.image_codes = image_codes
        self.mask = mask

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.image_codes[idx], self.mask[idx]

# Add triggering conditions
if __name__ == "__main__":
    # Create dummy data
    text = torch.randn(100, 128)
    image_codes = torch.randn(100, 128)
    mask = torch.randn(100, 128)

    # Create dataset and data loader
    dataset = TrainData(text, image_codes, mask)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create DALLE model and optimizer
    dalle = DALLE()
    opt = Adam(dalle.parameters(), lr=0.001)

    # Train DALLE model
    for batch in data_loader:
        text, image_codes, mask = batch
        # This line triggers the bug
        loss = dalle(text, image_codes, mask=mask)
        opt.zero_grad()
        loss.backward()
        opt.step()