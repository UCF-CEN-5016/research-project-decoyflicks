import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl

class YourModel(pl.LightningModule):
    def __init__(self, learning_rate=0.0003):
        super().__init__()
        self.learning_rate = learning_rate
        self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': self.scheduler, 'interval': 'epoch'}
        }

    def lr_scheduler_step(self, scheduler, optimizer_idx, epoch, batch_idx):
        pass  # Ensure this method is implemented to handle custom logic

# Example call when initializing the model
trainer = pl.Trainer(max_epochs=10, gpus=1)
model = YourModel(learning_rate=0.0003)
trainer.fit(model)