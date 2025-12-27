import torch
import torch.onnx as onnx
from fairseq import checkpoint_utils

hubert,_,_ = checkpoint_utils.load_model_ensemble_and_task(
    ["./assets/hubert/hubert_base.pt"],
    suffix="",
)
hubert_model = hubert[0]
hubert_model = hubert_model.half()

class HuberAdapter(torch.nn.Module):
    def __init__(self, model):
        super(HuberAdapter, self).__init__()
        self.model = model
    def forward(self,feats,padding_mask):
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 12
        }
        return self.model.extract_features(**inputs)

adapter = HuberAdapter(hubert_model)
feats = torch.randn(1, 100, 768)
padding_mask = torch.ones(1, 100, dtype=torch.int)

adapter = adapter.eval()
with torch.no_grad():
    torch.onnx.export(
        adapter,
        (feats,padding_mask),
        "hubert.onnx",
        input_names=["feats","padding_mask"],
        output_names=["logits","mask"],
        dynamic_axes={
            "feats": {0: "seq"},
            "padding_mask": {0: "seq"},
        }
    )