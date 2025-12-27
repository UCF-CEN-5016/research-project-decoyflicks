import torch
from fairseq import checkpoint_utils

# Load the pre-trained Hubert model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix=""
)
hubert_model = hubert[0]

# Convert model to half precision
hubert_model = hubert_model.half()

# Define a simple adapter for ONNX export
class HubertAdapter(torch.nn.Module):
    def __init__(self, model):
        super(HubertAdapter, self).__init__()
        self.model = model

    def forward(self, feats, padding_mask):
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 12
        }
        return self.model.extract_features(**inputs)

# Load input data
feats = torch.rand(1, 100, 100)  # Replace with real data
padding_mask = torch.zeros(1, 100, 100).bool()  # Replace with real data

# Create the adapter and export to ONNX
adapter = HubertAdapter(hubert_model)

# Export the model to ONNX format
try:
    torch.onnx.export(
        adapter,
        (feats, padding_mask),
        "hubert_model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["feats", "padding_mask"],
        output_names=["output"],
        dynamic_axes={
            "feats": {0: "batch_size", 1: "seq_len"},
            "padding_mask": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 1: "seq_len"}
        }
    )
except Exception as e:
    print(f"Error during ONNX export: {e}")