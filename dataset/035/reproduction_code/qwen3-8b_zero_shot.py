import torch

class RotaryPositionalEmbeddings:
    def __init__(self, dim):
        self.dim = dim

rotary_pe = RotaryPositionalEmbeddings(3)