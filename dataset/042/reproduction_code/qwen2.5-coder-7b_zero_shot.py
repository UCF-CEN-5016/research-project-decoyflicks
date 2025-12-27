from fairseq import checkpoint_utils
import torch


MODEL_PATH = "./assets/hubert/hubert_base.pt"
FEATS_PATH = "./feats.pt"
PADDING_MASK_PATH = "./padding_mask.pt"
OUTPUT_ONNX = "hubert.onnx"
OUTPUT_LAYER = 12


def load_hubert_model(path: str) -> torch.nn.Module:
    ensemble, _, _ = checkpoint_utils.load_model_ensemble_and_task([path], suffix="")
    model = ensemble[0]
    return model.half()


class HubertAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(HubertAdapter, self).__init__()
        self.model = model

    def forward(self, feats: torch.Tensor, padding_mask: torch.Tensor):
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": OUTPUT_LAYER,
        }
        return self.model.extract_features(**inputs)


def load_inputs(feats_path: str, padding_mask_path: str):
    feats = torch.load(feats_path)
    padding_mask = torch.load(padding_mask_path)
    return feats, padding_mask


def export_adapter_to_onnx(adapter: torch.nn.Module, feats: torch.Tensor, padding_mask: torch.Tensor, output_path: str):
    torch.onnx.export(
        adapter,
        (feats.cuda(), padding_mask.cuda()),
        output_path,
        input_names=["feats", "padding_mask"],
        output_names=["logits", "mask"],
        dynamic_axes={
            "feats": {0: "seq"},
            "padding_mask": {0: "seq"},
        },
    )


def main():
    model = load_hubert_model(MODEL_PATH)
    feats, padding_mask = load_inputs(FEATS_PATH, PADDING_MASK_PATH)
    adapter = HubertAdapter(model)
    export_adapter_to_onnx(adapter, feats, padding_mask, OUTPUT_ONNX)


if __name__ == "__main__":
    main()