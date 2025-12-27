import torch
import torch.onnx as onnx
from fairseq import checkpoint_utils

# Load model from fairseq checkpoint
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix=""
)
hubert_model = hubert[0]
hubert_model = hubert_model.half()  # Convert to float16

# Define adapter to wrap extract_features
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

# Load input tensors
feats = torch.load("./feats.pt")
padding_mask = torch.load("./padding_mask.pt")

# Attempt to export to ONNX
adapter = HubertAdapter(hubert_model)
feats_cuda, padding_mask_cuda = feats.cuda(), padding_mask.cuda()
onnx.export(
    adapter,
    (feats_cuda, padding_mask_cuda),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "seq"},
        "padding_mask": {0: "seq"}
    }
)