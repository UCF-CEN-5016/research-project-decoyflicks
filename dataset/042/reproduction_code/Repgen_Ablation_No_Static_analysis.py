import logging
from fairseq import checkpoint_utils

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("test")

# Load the Hubert model
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
hubert_model = hubert[0]
hubert_model.half()

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
feats = torch.load("./feats.pt").cuda()
padding_mask = torch.load("./padding_mask.pt").cuda()

# Exporting Adapter into ONNX with explicit static axes
torch.onnx.export(
    HuberAdapter(hubert_model),
    (feats, padding_mask),
    "hubert.onnx",
    input_names=["feats", "padding_mask"],
    output_names=["logits", "mask"],
    dynamic_axes={
        "feats": {0: "batch_size"},
        "padding_mask": {0: "batch_size"}
    },
)