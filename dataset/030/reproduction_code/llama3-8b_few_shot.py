import torch
from pytorch_lightning import Trainer, Module
from pytorch_lightning.callbacks.model_summary import ModelSummary

class CustomModel(Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        )

    def forward(self, x):
        return self.net(x)

model = CustomModel()

# Define a custom learning rate scheduler
class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    pass

scheduler = CosineAnnealingWarmRestarts(max_lr=0.1, warmup_steps=10)

trainer = Trainer(
    max_epochs=10,
    callbacks=[ModelSummary],
    optimizers=(torch.optim.SGD(model.parameters(), lr=0.001), None),
    scheduler=scheduler
)

# Train the model
trainer.fit(model)