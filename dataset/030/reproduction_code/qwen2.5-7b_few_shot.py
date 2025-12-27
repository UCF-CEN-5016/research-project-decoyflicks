import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer, LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        return [optimizer], [scheduler]

# Run the training
model = SimpleModel()
trainer = Trainer(max_epochs=1)
trainer.fit(model, DataLoader(TensorDataset(torch.randn(100, 10), torch.randn(100, 2)))