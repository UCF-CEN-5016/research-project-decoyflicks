import torch
from fairseq import checkpoint_utils

# Load the pretrained Hubert model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix=""
)
hubert_model = hubert[0].half()

# Create an adapter class for the Hubert model
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

# Load input features and padding mask
feats = torch.load("./feats.pt")
padding_mask = torch.load("./padding_mask.pt")

# Create an instance of the HubertAdapter
adapter = HubertAdapter(hubert_model)

# Export the adapter to ONNX format
adapter.eval()
feats_cuda, padding_mask_cuda = feats.cuda(), padding_mask.cuda()
torch.onnx.export(
    adapter,
    (feats_cuda, padding_mask_cuda),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "seq"},
        "padding_mask": {0: "seq"},
    }
)