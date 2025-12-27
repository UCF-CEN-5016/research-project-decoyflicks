import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class LitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

def main():
    model = LitModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0)

    trainer = Trainer(max_epochs=10, callbacks=[ModelSummary()])
    trainer.fit(model)

if __name__ == "__main__":
    main()