import torch
import torch.nn as nn
from fairseq.models.hubert import HubertModel

class DummyHubert(nn.Module):
    def extract_features(self, source, padding_mask, output_layer):
        return torch.randn(1, 10, 768), padding_mask

class HuberAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, feats, padding_mask):
        return self.model.extract_features(
            source=feats,
            padding_mask=padding_mask,
            output_layer=12
        )

hubert = DummyHubert()
adapter = HuberAdapter(hubert).half()
feats = torch.randn(1, 10, 768).half()
mask = torch.zeros(1, 10).bool()

torch.onnx.export(
    adapter,
    (feats.cuda(), mask.cuda()),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "seq"},
        "padding_mask": {0: "seq"},
    }
)