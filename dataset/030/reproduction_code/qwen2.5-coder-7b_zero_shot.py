import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from nnunet.utilities.nd_softmax import np_softmax

pl.seed_everything(42)


class SegmentationLitModel(pl.LightningModule):
    def __init__(self):
        super(SegmentationLitModel, self).__init__()
        self.net = ...

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        ...


def build_model():
    return SegmentationLitModel()


def build_data_module():
    return ...


def build_trainer(max_epochs=10, gpus=None):
    if gpus is None:
        gpus = [1]
    return pl.Trainer(max_epochs=max_epochs, gpus=gpus)


def main():
    model = build_model()
    data_module = build_data_module()
    trainer = build_trainer()
    trainer.fit(model, datamodule=data_module)


main()