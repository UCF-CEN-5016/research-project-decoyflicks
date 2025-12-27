import logging
from fairseq import checkpoint_utils
import torch

# Setup logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("learn_kmeans")

# Define HuberAdapter class
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

# Load the Hubert model and convert to half precision
hubert, _, _ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix=""
)
hubert_model = hubert[0]
hubert_model = hubert_model.half()

# Define input data
feats = torch.randn(10, 512).half()  # Example feature tensor
padding_mask = torch.zeros(10, dtype=torch.bool)  # Example padding mask

# Instantiate the adapter
adapter = HuberAdapter(hubert_model)

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
    print(f"Export failed with error: {e}")