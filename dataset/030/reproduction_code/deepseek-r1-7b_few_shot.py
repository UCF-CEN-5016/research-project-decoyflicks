import os
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Correctly initialize and use CosineAnnealingWarmRestarts scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer=your_optimizer,
    T_0=100,  # Initial number of epochs for cosine annealing
    T_mult=2,  # Multiplicative factor to increase epoch count each cycle
    eta_min=0.00001,  # Minimum learning rate
)

# Ensure the scheduler is correctly passed to the trainer
trainer = pl.Trainer(scheduler=scheduler)