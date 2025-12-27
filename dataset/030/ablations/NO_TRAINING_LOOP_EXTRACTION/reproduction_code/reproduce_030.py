import pytorch_lightning as pl
from nnunet import nnUNet
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class MyModel(pl.LightningModule):
    def __init__(self, learning_rate=0.0003):
        super(MyModel, self).__init__()
        self.model = nnUNet()  # Assuming nnUNet is properly defined
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]

if __name__ == "__main__":
    model = MyModel(learning_rate=0.0003)
    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(model)