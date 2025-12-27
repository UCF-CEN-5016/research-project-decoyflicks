import torch
from fairseq import checkpoint_utils

# Load pre-trained HuBERT model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)

# Create a HuBERT adapter for ONNX export
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

# Initialize the adapter and move to CUDA
hubert_model = hubert[0]
adapter = HuberAdapter(hubert_model)
adapter = adapter.half()
adapter = adapter.cuda()

# Sample input data
feats = torch.randn(1, 100, 256).half().cuda()
padding_mask = torch.ones(1, 100).half().cuda()

# Exporting Adapter into ONNX (this will cause the error)
torch.onnx.export(
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