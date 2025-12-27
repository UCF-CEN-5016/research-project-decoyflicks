import torch
from fairseq import checkpoint_utils

# Set up minimal Hubert model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],  # Replace with actual path
    suffix="",
)
hubert_model = hubert[0].half().cuda()  # Half precision triggers the issue

# Simple adapter class
class HubertAdapter(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, feats, padding_mask):
        return self.model.extract_features(
            source=feats,
            padding_mask=padding_mask,
            output_layer=12
        )

# Create dummy inputs (batch_size=1, seq_len=100, features=80)
feats = torch.randn(1, 100, 80).half().cuda()
padding_mask = torch.zeros(1, 100).bool().cuda()

# Attempt ONNX export
adapter = HubertAdapter(hubert_model)
try:
    torch.onnx.export(
        adapter,
        (feats, padding_mask),
        "hubert.onnx",
        input_names=["feats", "padding_mask"],
        output_names=["logits", "mask"],
        dynamic_axes={
            "feats": {0: "batch", 1: "seq"},
            "padding_mask": {0: "batch", 1: "seq"},
        }
    )
except Exception as e:
    print(f"Error during ONNX export: {e}")
    raise