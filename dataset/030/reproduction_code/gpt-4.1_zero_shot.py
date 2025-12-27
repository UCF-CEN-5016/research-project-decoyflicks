import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]

x = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=10)

model = LitModel()
trainer = pl.Trainer(max_epochs=2, accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=1 if torch.cuda.is_available() else None)
trainer.fit(model, loader)