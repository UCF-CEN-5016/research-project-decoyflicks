import torch
from fairseq import checkpoint_utils

# Load Hubert model (fairseq version 0.12.2)
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
model = hubert[0].half().cuda()
model.eval()

class HubertAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, feats, padding_mask):
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 12,
        }
        return self.model.extract_features(**inputs)

adapter = HubertAdapter(model)

# Dummy inputs with expected shapes and device
feats = torch.randn(1, 16000).half().cuda()        # Example input waveform
padding_mask = torch.zeros(1, 16000, dtype=torch.bool).cuda()

# ONNX export triggers AttributeError: 'Tensor' object has no attribute 'is_integer'
torch.onnx.export(
    adapter,
    (feats, padding_mask),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {1: "seq_len"},
        "padding_mask": {1: "seq_len"},
    },
    opset_version=14,
)