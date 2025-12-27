import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out.mean()
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]

model = MyModel()
trainer = pl.Trainer(accelerator='cpu', gpus=1)  # Updated 'devices' to 'gpus'
trainer.fit(model)