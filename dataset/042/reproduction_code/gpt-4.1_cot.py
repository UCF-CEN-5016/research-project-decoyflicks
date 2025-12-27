import torch
from fairseq import checkpoint_utils

# Load the Hubert model from checkpoint (make sure you have the checkpoint locally)
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
hubert_model = hubert[0]
hubert_model = hubert_model.half().cuda()

# Adapter wrapping the extract_features call
class HuberAdapter(torch.nn.Module):
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

adapter = HuberAdapter(hubert_model).cuda()

# Load dummy inputs (make sure these tensors are prepared beforehand)
feats = torch.load("./feats.pt").half().cuda()
padding_mask = torch.load("./padding_mask.pt").cuda()

# Trigger the bug: Export the adapter to ONNX
# This will raise: AttributeError: 'Tensor' object has no attribute 'is_integer'
torch.onnx.export(
    adapter,
    (feats, padding_mask),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "seq"},
        "padding_mask": {0: "seq"},
    },
)