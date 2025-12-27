import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from nnunet.utilities.nd_softmax import np_softmax

class LitModel(pl.LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.net = ...

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...

pl.seed_everything(42)
model = LitModel()
data_module = ...
trainer = pl.Trainer(max_epochs=10, gpus=[1])

def main():
    trainer.fit(model, datamodule=data_module)

main()