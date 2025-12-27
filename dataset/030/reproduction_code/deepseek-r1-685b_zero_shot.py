import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
        return [optimizer], [scheduler]

class DataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.randn(100, 10),
                torch.randint(0, 2, (100,))
            ),
            batch_size=10
        )

model = LitModel()
data = DataModule()
trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1)
trainer.fit(model, data)