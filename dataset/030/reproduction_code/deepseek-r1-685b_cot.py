import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
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

# Create synthetic data
data = [(torch.randn(32, 10), torch.randint(0, 2, (32,))) for _ in range(100)]
train_loader = torch.utils.data.DataLoader(data, batch_size=32)

# Initialize trainer and model
model = SimpleModel()
trainer = pl.Trainer(max_epochs=5, accelerator='gpu', devices=1)

# This will trigger the same error
trainer.fit(model, train_loader)

class FixedModel(SimpleModel):
    def lr_scheduler_step(self, scheduler, metric):
        # Custom handling for CosineAnnealingWarmRestarts
        if isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(self.current_epoch)  # or scheduler.step()
        else:
            scheduler.step(metric)

# Now this will work
fixed_model = FixedModel()
trainer.fit(fixed_model, train_loader)