import torch
from types import SimpleNamespace

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace()
        # Intentionally omit 'input_feat_per_channel'

model = DummyModel()
state = {'model': model}

try:
    print(state['model'].cfg.input_feat_per_channel)
except AttributeError as e:
    print(f"AttributeError: {e}")