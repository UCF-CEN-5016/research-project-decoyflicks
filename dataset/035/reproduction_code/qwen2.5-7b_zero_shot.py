import torch

class RotaryPositionalEmbeddings:
    def __init__(self, dim):
        self.dim = dim

# Create an instance of the RotaryPositionalEmbeddings class with dimension 3
rotary_pe = RotaryPositionalEmbeddings(3)