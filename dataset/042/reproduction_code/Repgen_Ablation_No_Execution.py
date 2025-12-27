import torch
from fairseq import checkpoint_utils

# Load Hubert model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
["./assets/hubert/hubert_base.pt"],
suffix=""
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

# Load input data
feats = torch.load("./feats.pt")
padding_mask = torch.load("./padding_mask.pt")

# Instantiate the adapter and ensure it is in the same device as input data (GPU here)
adapter = HuberAdapter(hubert_model).cuda()

try:
    # Exporting Adapter into ONNX
    torch.onnx.export(
        adapter,
        (feats.cuda(), padding_mask.cuda()),
        "hubert.onnx",
        input_names=["feats", "padding_mask"],
        output_names=["logits", "mask"],
        dynamic_axes={
            "feats": {0: "seq"},
            "padding_mask": {0: "seq"}
        }
    )
except Exception as e:
    print(f"Exporting to ONNX failed with the following error: {e}")