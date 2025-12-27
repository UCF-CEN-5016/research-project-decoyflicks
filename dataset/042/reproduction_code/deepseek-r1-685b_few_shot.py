import torch
from fairseq.models.hubert import HubertModel

# Bug Type: ONNX Export Failure
# Bug Description: AttributeError when converting hubert.extract_features() to ONNX
# Reproduction Code:

class HubertWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified Hubert-like model for reproduction
        self.conv = torch.nn.Conv1d(1, 64, kernel_size=10, stride=5)
        self.layer_norm = torch.nn.LayerNorm(64)
        
    def extract_features(self, source, padding_mask, output_layer):
        # Mock the extract_features behavior that fails in ONNX export
        x = self.conv(source)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.layer_norm(x)
        return x, padding_mask[:, :x.size(1)]

# Create test inputs
feats = torch.randn(1, 1, 1000)  # [batch, channel, seq_len]
padding_mask = torch.zeros(1, 1000, dtype=torch.bool)

# Attempt ONNX export
model = HubertWrapper().half().eval()
try:
    torch.onnx.export(
        model,
        (feats.cuda(), padding_mask.cuda(), 12),
        "hubert_test.onnx",
        input_names=["feats", "padding_mask", "output_layer"],
        output_names=["features", "out_padding_mask"],
        dynamic_axes={
            "feats": {2: "seq_len"},
            "padding_mask": {1: "seq_len"},
            "features": {1: "out_seq_len"},
            "out_padding_mask": {1: "out_seq_len"},
        },
        opset_version=14
    )
except Exception as e:
    print(f"ONNX export failed with error: {e}")