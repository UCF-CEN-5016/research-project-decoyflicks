import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

class LitModel(pl.LightningModule):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # important for this scheduler
                "frequency": 1,
            },
        }

# Dummy data loader
x = torch.randn(32, 10)
y = torch.randn(32, 1)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=8)

model = LitModel()

trainer = pl.Trainer(max_epochs=2, limit_train_batches=4, enable_checkpointing=False)
trainer.fit(model, train_dataloaders=loader)