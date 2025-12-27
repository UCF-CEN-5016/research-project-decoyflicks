import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelSummary
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Define a simple LightningModule
class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(5, 3)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = (self(x) - y).mean()
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return [optimizer], [scheduler]

# Set up the environment
if __name__ == "__main__":
    # Initialize the model, data, and trainer
    model = SimpleModel()
    data = torch.randn(100, 5), torch.randn(100, 3)
    trainer = Trainer(
        max_epochs=10,
        gpus=1,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        limit_test_batches=1.0,
        callbacks=[ModelSummary()],
    )

    # Trigger the bug
    trainer.fit(model, datamodule=None)