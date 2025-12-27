from fairseq import checkpoint_utils
import torch

# Step 1: Load the pre-trained Hubert model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
hubert_model = hubert[0]

# Step 2: Convert model to half precision
hubert_model = hubert_model.half()

# Step 3: Define a simple adapter for ONNX export
class HubertAdapter(torch.nn.Module):
    def __init__(self, model):
        super(HubertAdapter, self).__init__()
        self.model = model

    def forward(self, feats, padding_mask):
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 12  # This is an integer
        }
        return self.model.extract_features(**inputs)

# Step 4: Load input data (ensure these files exist in the correct path)
feats = torch.rand(1, 100, 100)  # Replace with real data
padding_mask = torch.zeros(1, 100, 100).bool()  # Replace with real data

# Step 5: Create the adapter and export to ONNX
adapter = HubertAdapter(hubert_model)

# Step 6: Export the model to ONNX format
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
            "output": {0: "batch_size", 1: "seq_len"},
        },
    )
except Exception as e:
    print(f"Error during ONNX export: {e}")