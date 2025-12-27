import logging
import os
import random
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

# Mocking imports for demonstration purposes
from utils import Accuracy

logger = logging.getLogger(__name__)

class DurationDataset(Dataset):
    def __init__(self, tsv_path, km_path, substring=""):
        # Mock implementation
        pass
    
    def __len__(self):
        return 100
    
    def __getitem__(self, i):
        x = torch.randint(0, 1000, (32,))
        y = torch.randint(0, 1000, (32,))
        return x, y

class CnnPredictor(nn.Module):
    def __init__(self, n_tokens, emb_dim, channels, kernel, output_dim, dropout, n_layers):
        super(CnnPredictor, self).__init__()
        # Mock implementation
        pass
    
    def forward(self, x):
        log_probs = torch.randn(32, 10, 1000)  # Simulated output with correct shape
        return log_probs

def train_epoch(model, loader, criterion, optimizer, device):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

# Mock data and initialization
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CnnPredictor(n_tokens=1000, emb_dim=256, channels=128, kernel=3, output_dim=1000, dropout=0.1, n_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_ds = DurationDataset(tsv_path="mock_tsv.tsv", km_path="mock_km.tsv")
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

# Run the train_epoch function
criterion = nn.CrossEntropyLoss()
train_epoch(model, train_dl, criterion=criterion, optimizer=optimizer, device=device)