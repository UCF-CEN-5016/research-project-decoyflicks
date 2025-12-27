import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelSummary

class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(5, 5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0003)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]

class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.x = torch.randn(100, 5)
        self.y = torch.randn(100, 5)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(list(zip(self.x, self.y)), batch_size=10)

model = Model()
data_module = DataModule()
logger = TensorBoardLogger('logs')
trainer = pl.Trainer(max_epochs=10, gpus=1, logger=logger, callbacks=[ModelSummary()])
trainer.fit(model, data_module)