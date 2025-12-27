import torch
import numpy as np

# Minimal setup
class MAE:
    def __init__(self):
        self.encoder = None

    def my_method(self, num_patches):
        # Triggering condition: Set up a sample input
        tokens = torch.tensor([[1, 2], [3, 4]])
        num_patches = 2

        # Bug-inducing code snippet
        pos_embedding = torch.tensor([[0.5, 0.2], [0.8, 0.6]])
        tokens = tokens + self.encoder.pos_embedding[:, 1:(num_patches + 1)]  # <--- BUGGY LINE

# Run the MAE and trigger the bug
mae = MAE()
mae.my_method(2)