import torch
from fairseq import checkpoint_utils
import torch.onnx as onnx

# Set up minimal environment
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Hubert model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
hubert_model = hubert[0]
hubert_model = hubert_model.half()
hubert_model.to(device)

# Create custom adapter class to wrap Hubert model
class HuberAdapter(torch.nn.Module):
    def __init__(self, model):
        super(HuberAdapter, self).__init__()
        self.model = model

    def forward(self, feats, padding_mask):
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 12
        }
        return self.model.extract_features(**inputs)

# Create adapter instance
adapter = HuberAdapter(hubert_model)

# Create input tensors
feats = torch.randn(1, 100, 768).to(device)
padding_mask = torch.ones(1, 100).to(device)

# Export adapter to ONNX format
onnx.export(
    adapter,
    (feats, padding_mask),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "seq"},
        "padding_mask": {0: "seq"},
    }
)