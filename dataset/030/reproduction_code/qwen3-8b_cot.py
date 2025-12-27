import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        out = self.layer(batch)
        loss = out.mean()
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_max=10)
        return optimizer, scheduler  # ❌ Returning a tuple instead of lists

model = MyModel()
trainer = pl.Trainer(accelerator='cpu', devices=1)
trainer.fit(model)

def configure_optimizers(self):
    optimizer = Adam(self.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_max=10)
    return [optimizer], [scheduler]