import torch
from pytorch_lightning import LightningModule, Trainer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class TestModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=0)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

# Main training loop
model = TestModel()
trainer = Trainer(gpus=1, max_epochs=10)
trainer.fit(model)