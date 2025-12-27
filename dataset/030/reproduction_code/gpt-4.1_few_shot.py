import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.01)
        # Using CosineAnnealingWarmRestarts scheduler without overriding lr_scheduler_step hook
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss',  # Optional but commonly set
        }

# Dummy dataset
class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        x = torch.randn(10)
        y = torch.randn(1)
        return x, y

dataset = DummyDataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=16)

model = SimpleModel()
trainer = pl.Trainer(max_epochs=3, devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu")

# This will raise MisconfigurationException about lr_scheduler API
trainer.fit(model, train_dataloaders=loader)