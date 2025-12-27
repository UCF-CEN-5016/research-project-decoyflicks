import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0003)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# Setup data and model
model = SimpleModel()
train_data = torch.utils.data.DataLoader(
    [(torch.randn(10), torch.randn(1)) for _ in range(100)],
    batch_size=10
)

# Trainer setup that reproduces the error
trainer = pl.Trainer(
    max_epochs=10,
    gpus=1,
    limit_train_batches=1.0,
    limit_val_batches=1.0
)

trainer.fit(model, train_dataloader=train_data)