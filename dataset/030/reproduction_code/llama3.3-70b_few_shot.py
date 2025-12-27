import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch

class NNUNetModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self(x), y)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.0003)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [lr_scheduler]

# Create a datamodule for demonstration purposes
class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data = torch.randn(100, 10), torch.randn(100, 10)

    def train_dataloader(self):
        return [(self.data[0][i:i+1], self.data[1][i:i+1]) for i in range(len(self.data[0]))]

# Initialize the model, data module, and trainer
model = NNUNetModel()
data_module = DataModule()
trainer = pl.Trainer(gpus=1, max_epochs=10)

# This will raise a MisconfigurationException
trainer.fit(model, data_module)