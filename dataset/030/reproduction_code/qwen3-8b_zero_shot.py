import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning import Trainer, LightningModule

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def training_step(self, batch, batch_idx):
        x = batch
        loss = F.mse_loss(self.layer(x), x)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

Trainer().fit(MyModel())

import torch
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning import Trainer, LightningModule

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def training_step(self, batch, batch_idx):
        x = batch
        loss = F.mse_loss(self.layer(x), x)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.1)
        # This is NOT a subclass of _LRScheduler
        scheduler = torch.optim.lr_scheduler._LRScheduler(optimizer)  # Invalid usage
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

Trainer().fit(MyModel())

import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning import Trainer, LightningModule

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def training_step(self, batch, batch_idx):
        x = batch
        loss = F.mse_loss(self.layer(x), x)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

Trainer().fit(MyModel())