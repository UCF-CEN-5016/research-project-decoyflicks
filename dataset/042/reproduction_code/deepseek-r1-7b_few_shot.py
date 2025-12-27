from fairseq import checkpoint_utils
import torch

# Load the model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
hubert_model = hubert[0]
hubert_model = hubert_model.half()

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

# Load and prepare input data
feats = torch.load("./feats.pt")
padding_mask = torch.load("./padding_mask.pt")

# Convert inputs to float16 and move to GPU
device = torch.device('cuda')
feats = feats.half().to(device)
padding_mask = padding_mask.half().to(device)

# Export the adapter into ONNX
torch.onnx.export(
    HuberAdapter(),
    (feats, padding_mask),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "seq"},
        "padding_mask": {0: "seq"},
    }
)