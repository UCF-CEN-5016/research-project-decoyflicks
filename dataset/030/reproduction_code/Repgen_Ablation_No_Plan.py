import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule, Trainer

class MyModel(LightningModule):
    def __init__(self, learning_rate: float = 0.0003, **kwargs):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = TestMod()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=5e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def training_step(self, batch, batch_idx):
        x, y = batch
        out1 = self.model(x)
        loss = torch.nn.CrossEntropyLoss()(out1, y)
        return loss

    def lr_scheduler_step(self, scheduler, optimizer_index, epoch):
        # Override this method to handle custom scheduling logic if needed
        pass

class TestMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )

    def forward(self, inputs):
        return self.layers(inputs)

# Example usage
if __name__ == "__main__":
    model = MyModel()
    dataloader = ...  # Define your DataLoader here
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, dataloader)