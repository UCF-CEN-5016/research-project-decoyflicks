import torch
from fairseq import checkpoint_utils

hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./hubert_base.pt"],
    suffix="",
)
model = hubert[0].half()

class Adapter(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, feats, padding_mask):
        return self.m.extract_features(source=feats, padding_mask=padding_mask, output_layer=12)

adapter = Adapter(model)

feats = torch.randn(1, 16000).half()
padding_mask = torch.zeros(1, 16000, dtype=torch.bool)

torch.onnx.export(
    adapter,
    (feats.cuda(), padding_mask.cuda()),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={"feats": {0: "seq"}, "padding_mask": {0: "seq"}},
)