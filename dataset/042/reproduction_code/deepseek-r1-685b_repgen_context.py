import torch

class HubertWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 64, kernel_size=10, stride=5)
        self.layer_norm = torch.nn.LayerNorm(64)
        
    def extract_features(self, source, padding_mask):
        x = self.conv(source)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return x, padding_mask[:, :x.size(1)]

# Create test inputs
feats = torch.randn(1, 1, 1000)
padding_mask = torch.zeros(1, 1000, dtype=torch.bool)

# Attempt ONNX export
model = HubertWrapper().half().eval()
try:
    example_input = (feats.cuda(), padding_mask.cuda())
    torch.onnx.export(
        model.extract_features,
        example_input,
        "hubert_test.onnx",
        input_names=["feats", "padding_mask"],
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